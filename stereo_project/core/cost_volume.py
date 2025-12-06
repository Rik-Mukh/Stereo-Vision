"""
Cost volume construction for stereo matching.

Implements Sum of Absolute Differences (SAD) and Census transform costs using
NumPy only. These volumes feed into SGM or other optimizers.
"""

import numpy as np

# Lookup table for counting bits in bytes (0-255)
_BIT_COUNT_LUT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def _box_filter_sum(image: np.ndarray, radius: int) -> np.ndarray:
    """
    Sum over a (2r+1)x(2r+1) window using an integral image for efficiency.
    """
    if radius == 0:
        return image
    pad = np.pad(image, ((radius, radius), (radius, radius)), mode="constant")
    cs = pad.cumsum(axis=0).cumsum(axis=1)
    # area sum via inclusion-exclusion on the integral image
    win = (
        cs[2 * radius :, 2 * radius :]
        - cs[:-2 * radius, 2 * radius :]
        - cs[2 * radius :, :-2 * radius]
        + cs[:-2 * radius, :-2 * radius]
    )
    return win


def compute_sad_cost_volume(
    left: np.ndarray,
    right: np.ndarray,
    max_disparity: int,
    window_size: int = 5,
    right_to_left: bool = False,
) -> np.ndarray:
    """
    Build SAD cost volume C[y, x, d] with d in [0, max_disparity).

    left, right are rectified grayscale float32 images in [0, 1].
    
    Parameters
    ----------
    right_to_left : bool
        If False (default): left-to-right matching. For pixel (x, y) in left,
            find match at (x - d, y) in right. Shifts right image RIGHT by d.
        If True: right-to-left matching. For pixel (x, y) in right,
            find match at (x + d, y) in left. Shifts left image LEFT by d.
    """
    if left.shape != right.shape:
        raise ValueError("Left and right images must have the same shape.")
    if left.ndim != 2:
        raise ValueError("Inputs must be single-channel grayscale images.")
    H, W = left.shape
    radius = window_size // 2
    cost_volume = np.zeros((H, W, max_disparity), dtype=np.float32)

    for d in range(max_disparity):
        if d == 0:
            if right_to_left:
                shifted = left
            else:
                shifted = right
        else:
            if right_to_left:
                # Right-to-left: shift left image LEFT by d so left[x+d] aligns with right[x]
                shifted = np.zeros_like(left)
                shifted[:, :-d] = left[:, d:]
            else:
                # Left-to-right: shift right image RIGHT by d so right[x-d] aligns with left[x]
                shifted = np.zeros_like(right)
                shifted[:, d:] = right[:, :-d]

        if right_to_left:
            abs_diff = np.abs(right - shifted).astype(np.float32)
        else:
            abs_diff = np.abs(left - shifted).astype(np.float32)
        cost = _box_filter_sum(abs_diff, radius)
        cost_volume[:, :, d] = cost

    return cost_volume


def census_transform(image: np.ndarray, window_size: int = 7) -> np.ndarray:
    """
    Compute the Census transform.

    For each pixel, compares neighbors within the window to the center pixel
    and produces a bitstring; robust to illumination changes since it encodes
    relative ordering instead of absolute intensity.
    """
    if image.ndim != 2:
        raise ValueError("Census expects a single-channel image.")
    H, W = image.shape
    radius = window_size // 2
    bits = window_size * window_size - 1

    pad = np.pad(image, ((radius, radius), (radius, radius)), mode="constant")
    center = pad[radius : radius + H, radius : radius + W]

    if bits <= 64:
        desc = np.zeros((H, W), dtype=np.uint64)
        bit_idx = 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy == 0 and dx == 0:
                    continue
                neighbor = pad[radius + dy : radius + dy + H, radius + dx : radius + dx + W]
                # Using < so equal intensities yield 0; choice is arbitrary but consistent.
                bit = (neighbor < center).astype(np.uint64)
                desc |= (bit << bit_idx)
                bit_idx += 1
    else:
        # Fallback: store each comparison as a separate bit in uint8 for clarity.
        desc = np.zeros((H, W, bits), dtype=np.uint8)
        bit_idx = 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy == 0 and dx == 0:
                    continue
                neighbor = pad[radius + dy : radius + dy + H, radius + dx : radius + dx + W]
                desc[:, :, bit_idx] = (neighbor < center).astype(np.uint8)
                bit_idx += 1

    return desc


def _hamming_distance_u64(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Hamming distance for packed uint64 Census descriptors.
    
    Counts set bits using a lookup table for each byte, then sums across bytes.
    """
    xor = np.bitwise_xor(a, b)
    # Extract bytes from uint64: view as uint8 array
    xor_bytes = xor.view(np.uint8)
    # Count bits in each byte using lookup table
    bit_counts = _BIT_COUNT_LUT[xor_bytes]
    # Sum across the 8 bytes of each uint64
    hamming = bit_counts.reshape(a.shape + (8,)).sum(axis=-1).astype(np.float32)
    return hamming


def compute_census_cost_volume(
    left: np.ndarray,
    right: np.ndarray,
    max_disparity: int,
    window_size: int = 7,
    right_to_left: bool = False,
) -> np.ndarray:
    """
    Build Census-based cost volume C[y, x, d] with d in [0, max_disparity).

    Cost is Hamming distance between Census descriptors. Census is robust to
    illumination changes because it compares local ordering rather than raw intensity.
    
    Parameters
    ----------
    right_to_left : bool
        If False (default): left-to-right matching. For pixel (x, y) in left,
            find match at (x - d, y) in right. Shifts right descriptor RIGHT by d.
        If True: right-to-left matching. For pixel (x, y) in right,
            find match at (x + d, y) in left. Shifts left descriptor LEFT by d.
    """
    if left.shape != right.shape:
        raise ValueError("Left and right images must have the same shape.")
    if left.ndim != 2:
        raise ValueError("Inputs must be single-channel grayscale images.")

    H, W = left.shape
    left_desc = census_transform(left, window_size)
    right_desc = census_transform(right, window_size)

    cost_volume = np.zeros((H, W, max_disparity), dtype=np.float32)
    packed = left_desc.dtype == np.uint64

    for d in range(max_disparity):
        if d == 0:
            if right_to_left:
                shifted_desc = left_desc
                ref_desc = right_desc
            else:
                shifted_desc = right_desc
                ref_desc = left_desc
        else:
            if right_to_left:
                # Right-to-left: shift left descriptor LEFT by d
                ref_desc = right_desc
                if packed:
                    shifted_desc = np.zeros_like(left_desc)
                    shifted_desc[:, :-d] = left_desc[:, d:]
                else:
                    shifted_desc = np.zeros_like(left_desc)
                    shifted_desc[:, :-d, :] = left_desc[:, d:, :]
            else:
                # Left-to-right: shift right descriptor RIGHT by d
                ref_desc = left_desc
                if packed:
                    shifted_desc = np.zeros_like(right_desc)
                    shifted_desc[:, d:] = right_desc[:, :-d]
                else:
                    shifted_desc = np.zeros_like(right_desc)
                    shifted_desc[:, d:, :] = right_desc[:, :-d, :]

        if packed:
            cost = _hamming_distance_u64(ref_desc, shifted_desc)
        else:
            # Cast to int16 before subtraction to avoid uint8 underflow (0-1 = 255)
            diff = ref_desc.astype(np.int16) - shifted_desc.astype(np.int16)
            cost = np.sum(np.abs(diff), axis=2).astype(np.float32)

        cost_volume[:, :, d] = cost

    return cost_volume


def normalize_cost_volume(cost_volume: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize cost volume to improve discrimination and SGM performance.
    
    Parameters
    ----------
    cost_volume : np.ndarray
        Raw cost volume of shape (H, W, D).
    method : str
        Normalization method:
        - "minmax": Per-pixel min-max normalization to [0, 1]
        - "zscore": Per-pixel z-score normalization
        - "global": Global min-max normalization
    
    Returns
    -------
    np.ndarray
        Normalized cost volume.
    """
    if method == "minmax":
        # Per-pixel min-max normalization
        c_min = cost_volume.min(axis=2, keepdims=True)
        c_max = cost_volume.max(axis=2, keepdims=True)
        denom = c_max - c_min
        denom = np.where(denom == 0, 1.0, denom)  # Avoid division by zero
        normalized = (cost_volume - c_min) / denom
    elif method == "zscore":
        # Per-pixel z-score normalization
        c_mean = cost_volume.mean(axis=2, keepdims=True)
        c_std = cost_volume.std(axis=2, keepdims=True)
        c_std = np.where(c_std == 0, 1.0, c_std)  # Avoid division by zero
        normalized = (cost_volume - c_mean) / c_std
    elif method == "global":
        # Global min-max normalization
        c_min = cost_volume.min()
        c_max = cost_volume.max()
        if c_max - c_min > 0:
            normalized = (cost_volume - c_min) / (c_max - c_min)
        else:
            normalized = cost_volume
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized.astype(np.float32)


def compute_confidence(cost_volume: np.ndarray, method: str = "peak_ratio") -> np.ndarray:
    """
    Compute per-pixel confidence from the cost volume.
    
    High confidence indicates a clear minimum (good match).
    Low confidence indicates ambiguous matching.
    
    Parameters
    ----------
    cost_volume : np.ndarray
        Cost volume of shape (H, W, D).
    method : str
        Confidence computation method:
        - "peak_ratio": Ratio of second-best to best cost (higher = more confident)
        - "curvature": Second derivative at minimum (higher = sharper peak)
        - "winner_margin": Difference between best and second-best
    
    Returns
    -------
    np.ndarray
        Confidence map of shape (H, W) in [0, 1].
    """
    H, W, D = cost_volume.shape
    
    # Find best disparity and its cost
    best_d = np.argmin(cost_volume, axis=2)
    best_cost = np.min(cost_volume, axis=2)
    
    if method == "peak_ratio":
        # Find second-best cost by masking out the best
        ys, xs = np.indices((H, W))
        costs_masked = cost_volume.copy()
        costs_masked[ys, xs, best_d] = np.inf
        second_cost = np.min(costs_masked, axis=2)
        
        # Ratio: closer to 1 means ambiguous, closer to 0 means confident
        # Invert so higher = more confident
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = best_cost / second_cost
        ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)
        confidence = 1.0 - ratio  # Higher when best << second
        confidence = np.clip(confidence, 0, 1)
        
    elif method == "curvature":
        # Second derivative at minimum: (c[d-1] + c[d+1] - 2*c[d]) / 1^2
        confidence = np.zeros((H, W), dtype=np.float32)
        for y in range(H):
            for x in range(W):
                d = int(best_d[y, x])
                if 0 < d < D - 1:
                    c_m1 = cost_volume[y, x, d - 1]
                    c_0 = cost_volume[y, x, d]
                    c_p1 = cost_volume[y, x, d + 1]
                    curv = c_m1 + c_p1 - 2 * c_0
                    confidence[y, x] = curv
                else:
                    confidence[y, x] = 0
        # Normalize to [0, 1]
        c_max = confidence.max()
        if c_max > 0:
            confidence = confidence / c_max
            
    elif method == "winner_margin":
        # Absolute margin between best and second-best
        ys, xs = np.indices((H, W))
        costs_masked = cost_volume.copy()
        costs_masked[ys, xs, best_d] = np.inf
        second_cost = np.min(costs_masked, axis=2)
        margin = second_cost - best_cost
        
        # Normalize to [0, 1]
        m_max = margin.max()
        if m_max > 0:
            confidence = margin / m_max
        else:
            confidence = np.zeros((H, W), dtype=np.float32)
    else:
        raise ValueError(f"Unknown confidence method: {method}")
    
    return confidence.astype(np.float32)

