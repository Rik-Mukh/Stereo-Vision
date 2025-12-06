"""
Disparity refinement utilities (subpixel interpolation, consistency checks,
speckle filtering, and simple hole filling).
"""

import warnings

import numpy as np


def subpixel_refine_disparity(
    aggregated_costs: np.ndarray,
    integer_disparity: np.ndarray,
) -> np.ndarray:
    """
    Subpixel refinement via quadratic (parabolic) fit around d-1, d, d+1.
    """
    if aggregated_costs.ndim != 3:
        raise ValueError("aggregated_costs must have shape (H, W, D)")
    if aggregated_costs.shape[:2] != integer_disparity.shape:
        raise ValueError("integer_disparity shape must match image size.")

    H, W, D = aggregated_costs.shape
    disp = integer_disparity.astype(np.float32, copy=True)

    # Only refine where we have neighbors on both sides.
    mask = (integer_disparity > 0) & (integer_disparity < D - 1)
    ys, xs = np.nonzero(mask)

    for y, x in zip(ys, xs):
        d = int(integer_disparity[y, x])
        c_m1 = aggregated_costs[y, x, d - 1]
        c_0 = aggregated_costs[y, x, d]
        c_p1 = aggregated_costs[y, x, d + 1]
        denom = 2 * (c_m1 - 2 * c_0 + c_p1)
        if denom == 0:
            continue
        delta = (c_m1 - c_p1) / denom  # vertex of the parabola
        disp[y, x] = d + np.clip(delta, -1.0, 1.0)  # clamp to avoid huge jumps

    return disp


def left_right_consistency_check(
    disp_left: np.ndarray,
    disp_right: np.ndarray,
    max_diff: float = 1.0,
) -> np.ndarray:
    """
    Validate left disparities using the right disparity map.

    Returns
    -------
    mask : np.ndarray of bool
        True where |d_L(x, y) - d_R(x - d_L, y)| <= max_diff, else False.
    Invalid/out-of-bounds lookups are marked False.
    """
    if disp_left.shape != disp_right.shape:
        raise ValueError("Left and right disparity maps must match in shape.")

    H, W = disp_left.shape
    mask = np.ones_like(disp_left, dtype=bool)

    ys, xs = np.indices((H, W))
    xr = xs - np.round(disp_left).astype(int)
    valid_coords = (xr >= 0) & (xr < W)
    mask &= valid_coords

    xr_clip = np.clip(xr, 0, W - 1)
    dr = disp_right[ys, xr_clip]
    diff = np.abs(disp_left - dr)
    mask &= diff <= max_diff
    return mask


def confidence_filter(
    disparity: np.ndarray,
    confidence: np.ndarray,
    threshold: float = 0.3,
    invalid_value: float = 0.0,
) -> np.ndarray:
    """
    Filter disparity based on confidence map.
    
    Low-confidence pixels are set to invalid_value.
    
    Parameters
    ----------
    disparity : np.ndarray
        Disparity map of shape (H, W).
    confidence : np.ndarray
        Confidence map of shape (H, W) in [0, 1].
    threshold : float
        Minimum confidence threshold. Pixels below this are invalidated.
    invalid_value : float
        Value to assign to low-confidence pixels.
    
    Returns
    -------
    np.ndarray
        Filtered disparity map.
    """
    filtered = disparity.copy()
    low_conf = confidence < threshold
    filtered[low_conf] = invalid_value
    return filtered


def speckle_filter(
    disparity: np.ndarray,
    valid_mask: np.ndarray | None = None,
    max_speckle_size: int = 100,
) -> np.ndarray:
    """
    Remove small isolated regions (speckles) using 8-connected components.

    Pixels with disparity == 0 are treated as invalid. Regions smaller than
    max_speckle_size are set to 0.
    """
    H, W = disparity.shape
    disp = disparity.copy()
    if valid_mask is None:
        valid = disp > 0
    else:
        valid = (disp > 0) & valid_mask

    visited = np.zeros_like(valid, dtype=bool)
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for y in range(H):
        for x in range(W):
            if visited[y, x] or not valid[y, x]:
                continue
            stack = [(y, x)]
            component = []
            visited[y, x] = True

            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx] and valid[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))

            if len(component) < max_speckle_size:
                for cy, cx in component:
                    disp[cy, cx] = 0.0

    return disp


def fill_invalid_disparities(disparity: np.ndarray, invalid_value: float = 0.0) -> np.ndarray:
    """
    Fill invalid pixels (== invalid_value) with simple scanline interpolation.
    This is a basic fallback; prefer fill_invalid_bilateral for better results.
    """
    H, W = disparity.shape
    disp = disparity.copy().astype(np.float32)

    # Left-to-right and right-to-left nearest valid disparities.
    left_fill = np.full_like(disp, np.nan, dtype=np.float32)
    right_fill = np.full_like(disp, np.nan, dtype=np.float32)

    for y in range(H):
        last = np.nan
        for x in range(W):
            if disp[y, x] != invalid_value:
                last = disp[y, x]
            left_fill[y, x] = last
        last = np.nan
        for x in range(W - 1, -1, -1):
            if disp[y, x] != invalid_value:
                last = disp[y, x]
            right_fill[y, x] = last

    filled = disp.copy()
    invalid = disp == invalid_value
    
    # Compute mean of left and right fills, suppressing warning for all-NaN slices
    stacked = np.stack([left_fill, right_fill], axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lr_avg = np.nanmean(stacked, axis=0)
    # For pixels where both are NaN, keep invalid_value
    filled[invalid] = np.nan_to_num(lr_avg[invalid], nan=invalid_value)
    return filled


def fill_invalid_bilateral(
    disparity: np.ndarray,
    guide_image: np.ndarray | None = None,
    invalid_value: float = 0.0,
    window_size: int = 5,
    sigma_space: float = 5.0,
    sigma_color: float = 0.1,
    max_iterations: int = 5,
) -> np.ndarray:
    """
    Fill invalid disparity pixels using iterative bilateral-like weighted averaging.
    
    Uses spatial distance and optionally image intensity to guide the filling.
    Invalid pixels are filled from nearby valid pixels using Gaussian weights.
    
    Parameters
    ----------
    disparity : np.ndarray
        Disparity map with invalid pixels marked as invalid_value.
    guide_image : np.ndarray, optional
        Grayscale guide image for edge-aware filling. If None, only spatial
        weighting is used.
    invalid_value : float
        Value marking invalid disparity pixels.
    window_size : int
        Size of the window for weighted averaging (must be odd).
    sigma_space : float
        Standard deviation for spatial Gaussian weighting.
    sigma_color : float
        Standard deviation for intensity-based Gaussian weighting (if guide provided).
    max_iterations : int
        Number of iterations to propagate valid disparities into invalid regions.
    """
    H, W = disparity.shape
    disp = disparity.copy().astype(np.float32)
    radius = window_size // 2
    
    # Precompute spatial Gaussian weights
    y_off, x_off = np.mgrid[-radius:radius+1, -radius:radius+1]
    spatial_weights = np.exp(-(x_off**2 + y_off**2) / (2 * sigma_space**2))
    
    for _ in range(max_iterations):
        invalid = disp == invalid_value
        if not invalid.any():
            break
        
        # Pad arrays for windowed access
        disp_pad = np.pad(disp, radius, mode='reflect')
        valid_pad = np.pad(~invalid, radius, mode='constant', constant_values=False)
        
        if guide_image is not None:
            guide_pad = np.pad(guide_image, radius, mode='reflect')
        
        new_disp = disp.copy()
        
        # Process invalid pixels
        ys, xs = np.nonzero(invalid)
        for y, x in zip(ys, xs):
            # Extract local window
            win_disp = disp_pad[y:y+window_size, x:x+window_size]
            win_valid = valid_pad[y:y+window_size, x:x+window_size]
            
            if not win_valid.any():
                continue
            
            # Compute weights
            weights = spatial_weights.copy()
            
            if guide_image is not None:
                center_val = guide_image[y, x]
                win_guide = guide_pad[y:y+window_size, x:x+window_size]
                color_weights = np.exp(-((win_guide - center_val)**2) / (2 * sigma_color**2))
                weights = weights * color_weights
            
            # Only use valid neighbors
            weights = weights * win_valid.astype(np.float32)
            weight_sum = weights.sum()
            
            if weight_sum > 0:
                new_disp[y, x] = (weights * win_disp).sum() / weight_sum
        
        disp = new_disp
    
    return disp


def median_filter_disparity(
    disparity: np.ndarray,
    kernel_size: int = 5,
    invalid_value: float | None = None,
) -> np.ndarray:
    """
    Apply median filter to reduce noise and horizontal streaks in disparity map.
    
    Parameters
    ----------
    disparity : np.ndarray
        Input disparity map.
    kernel_size : int
        Size of the median filter kernel (must be odd).
    invalid_value : float, optional
        If provided, invalid pixels are excluded from the median computation
        and preserved in the output.
    
    Returns
    -------
    np.ndarray
        Filtered disparity map.
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    
    H, W = disparity.shape
    radius = kernel_size // 2
    disp = disparity.astype(np.float32)
    
    # Pad the disparity map
    disp_pad = np.pad(disp, radius, mode='reflect')
    filtered = np.zeros_like(disp)
    
    for y in range(H):
        for x in range(W):
            window = disp_pad[y:y+kernel_size, x:x+kernel_size].flatten()
            
            if invalid_value is not None:
                # Exclude invalid values from median
                valid_vals = window[window != invalid_value]
                if len(valid_vals) > 0:
                    filtered[y, x] = np.median(valid_vals)
                else:
                    filtered[y, x] = invalid_value
            else:
                filtered[y, x] = np.median(window)
    
    return filtered


def weighted_median_filter(
    disparity: np.ndarray,
    guide_image: np.ndarray,
    kernel_size: int = 5,
    sigma_space: float = 5.0,
    sigma_color: float = 0.1,
) -> np.ndarray:
    """
    Edge-preserving weighted median filter guided by image intensity.
    
    Uses bilateral weights (spatial + intensity) to compute a weighted median,
    which preserves edges better than a standard median filter.
    
    Parameters
    ----------
    disparity : np.ndarray
        Input disparity map.
    guide_image : np.ndarray
        Grayscale guide image (same size as disparity).
    kernel_size : int
        Size of the filter kernel (must be odd).
    sigma_space : float
        Standard deviation for spatial Gaussian weighting.
    sigma_color : float
        Standard deviation for intensity-based Gaussian weighting.
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    
    H, W = disparity.shape
    radius = kernel_size // 2
    
    # Precompute spatial weights
    y_off, x_off = np.mgrid[-radius:radius+1, -radius:radius+1]
    spatial_weights = np.exp(-(x_off**2 + y_off**2) / (2 * sigma_space**2))
    
    # Pad arrays
    disp_pad = np.pad(disparity.astype(np.float32), radius, mode='reflect')
    guide_pad = np.pad(guide_image.astype(np.float32), radius, mode='reflect')
    
    filtered = np.zeros((H, W), dtype=np.float32)
    
    for y in range(H):
        for x in range(W):
            # Extract windows
            win_disp = disp_pad[y:y+kernel_size, x:x+kernel_size]
            win_guide = guide_pad[y:y+kernel_size, x:x+kernel_size]
            center_val = guide_image[y, x]
            
            # Compute bilateral weights
            color_weights = np.exp(-((win_guide - center_val)**2) / (2 * sigma_color**2))
            weights = spatial_weights * color_weights
            
            # Compute weighted median
            flat_disp = win_disp.flatten()
            flat_weights = weights.flatten()
            
            # Sort by disparity and find weighted median
            sorted_idx = np.argsort(flat_disp)
            sorted_disp = flat_disp[sorted_idx]
            sorted_weights = flat_weights[sorted_idx]
            
            cumsum = np.cumsum(sorted_weights)
            total = cumsum[-1]
            median_idx = np.searchsorted(cumsum, total / 2)
            filtered[y, x] = sorted_disp[median_idx]
    
    return filtered


def bilateral_filter_disparity(
    disparity: np.ndarray,
    guide_image: np.ndarray,
    kernel_size: int = 7,
    sigma_space: float = 7.0,
    sigma_color: float = 0.1,
    sigma_disp: float = 5.0,
) -> np.ndarray:
    """
    Edge-preserving bilateral filter for disparity smoothing.
    
    Uses joint bilateral filtering with both image intensity and disparity
    similarity to smooth while preserving edges.
    
    Parameters
    ----------
    disparity : np.ndarray
        Input disparity map.
    guide_image : np.ndarray
        Grayscale guide image (same size as disparity).
    kernel_size : int
        Size of the filter kernel (must be odd).
    sigma_space : float
        Standard deviation for spatial Gaussian weighting.
    sigma_color : float
        Standard deviation for intensity-based Gaussian weighting.
    sigma_disp : float
        Standard deviation for disparity-based weighting (to avoid smoothing
        across depth discontinuities).
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    
    H, W = disparity.shape
    radius = kernel_size // 2
    
    # Precompute spatial weights
    y_off, x_off = np.mgrid[-radius:radius+1, -radius:radius+1]
    spatial_weights = np.exp(-(x_off**2 + y_off**2) / (2 * sigma_space**2))
    
    # Pad arrays
    disp_pad = np.pad(disparity.astype(np.float32), radius, mode='reflect')
    guide_pad = np.pad(guide_image.astype(np.float32), radius, mode='reflect')
    
    filtered = np.zeros((H, W), dtype=np.float32)
    
    for y in range(H):
        for x in range(W):
            # Extract windows
            win_disp = disp_pad[y:y+kernel_size, x:x+kernel_size]
            win_guide = guide_pad[y:y+kernel_size, x:x+kernel_size]
            center_color = guide_image[y, x]
            center_disp = disparity[y, x]
            
            # Compute bilateral weights
            color_weights = np.exp(-((win_guide - center_color)**2) / (2 * sigma_color**2))
            disp_weights = np.exp(-((win_disp - center_disp)**2) / (2 * sigma_disp**2))
            weights = spatial_weights * color_weights * disp_weights
            
            # Weighted average
            weight_sum = weights.sum()
            if weight_sum > 0:
                filtered[y, x] = (weights * win_disp).sum() / weight_sum
            else:
                filtered[y, x] = center_disp
    
    return filtered


# Backward-compatible aliases for earlier naming.
def subpixel_refine(disparity_map: np.ndarray, cost_volume: np.ndarray) -> np.ndarray:  # pragma: no cover - legacy
    integer_disp = np.argmin(cost_volume, axis=2)
    return subpixel_refine_disparity(cost_volume, integer_disp)
