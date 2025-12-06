"""
Semi-Global Matching (SGM) optimizer.

Aggregates matching costs along multiple 1D paths with smoothness penalties
(P1 for small disparity changes, P2 for larger jumps). SGM approximates a
global MRF energy (data + smoothness) with tractable dynamic programming.
"""

import numpy as np


def _traversal_ranges(H: int, W: int, dy: int, dx: int):
    """
    Generate scan order so predecessors (p - r) are visited before p.
    """
    y_range = range(H) if dy >= 0 else range(H - 1, -1, -1)
    x_range = range(W) if dx >= 0 else range(W - 1, -1, -1)
    return y_range, x_range


def compute_adaptive_P2(
    image: np.ndarray,
    P2_base: float = 8.0,
    P2_min: float = 2.0,
    gradient_scale: float = 10.0,
) -> np.ndarray:
    """
    Compute adaptive P2 penalty based on image gradients.
    
    At strong edges (high gradient), reduce P2 to allow disparity discontinuities.
    In smooth regions (low gradient), use full P2 to encourage smooth disparity.
    
    P2(p) = max(P2_min, P2_base / (1 + gradient_scale * |∇I(p)|))
    
    Parameters
    ----------
    image : np.ndarray
        Grayscale image (H, W) in [0, 1].
    P2_base : float
        Base P2 penalty for smooth regions.
    P2_min : float
        Minimum P2 penalty at strong edges.
    gradient_scale : float
        Scaling factor for gradient influence.
    
    Returns
    -------
    np.ndarray
        Per-pixel P2 values of shape (H, W).
    """
    # Compute gradient magnitude using Sobel-like differences
    grad_x = np.abs(np.diff(image, axis=1, prepend=image[:, :1]))
    grad_y = np.abs(np.diff(image, axis=0, prepend=image[:1, :]))
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Adaptive P2: reduce at edges
    P2_adaptive = P2_base / (1.0 + gradient_scale * grad_mag)
    P2_adaptive = np.maximum(P2_adaptive, P2_min)
    
    return P2_adaptive.astype(np.float32)


def semi_global_matching(
    cost_volume: np.ndarray,
    P1: float = 0.8,
    P2: float | np.ndarray = 8.0,
    directions: tuple[tuple[int, int], ...] | None = None,
) -> np.ndarray:
    """
    Perform SGM cost aggregation.

    cost_volume: float32 array (H, W, D), disparities d in [0, D).
    P1: Small penalty for disparity change of 1.
    P2: Large penalty for disparity change > 1. Can be a scalar or per-pixel array (H, W).
    """
    if directions is None:
        directions = (
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )

    if cost_volume.ndim != 3:
        raise ValueError("Cost volume must be (H, W, D).")

    H, W, D = cost_volume.shape
    C = cost_volume.astype(np.float32, copy=False)
    aggregated = np.zeros_like(C, dtype=np.float32)
    
    # Handle scalar vs adaptive P2
    P2_is_adaptive = isinstance(P2, np.ndarray)
    if P2_is_adaptive and P2.shape != (H, W):
        raise ValueError(f"Adaptive P2 must have shape ({H}, {W}), got {P2.shape}")

    for dy, dx in directions:
        L = np.zeros_like(C, dtype=np.float32)
        y_range, x_range = _traversal_ranges(H, W, dy, dx)

        for y in y_range:
            for x in x_range:
                py, px = y - dy, x - dx
                if py < 0 or py >= H or px < 0 or px >= W:
                    # Border: no predecessor along this path.
                    L[y, x, :] = C[y, x, :]
                    continue

                prev = L[py, px, :]
                prev_min = prev.min()
                
                # Get P2 for this pixel (scalar or from adaptive map)
                P2_val = P2[y, x] if P2_is_adaptive else P2

                # Costs for same disparity, disparity-1, disparity+1, and any other.
                same = prev
                d_minus = np.roll(prev, 1) + P1
                d_minus[0] = np.inf
                d_plus = np.roll(prev, -1) + P1
                d_plus[-1] = np.inf
                others = prev_min + P2_val

                best_prev = np.minimum(np.minimum(same, d_minus), np.minimum(d_plus, others))

                L[y, x, :] = C[y, x, :] + best_prev - prev_min

        aggregated += L

    return aggregated


def winner_take_all(aggregated_costs: np.ndarray) -> np.ndarray:
    """
    Choose disparity with minimal aggregated cost (WTA).
    """
    if aggregated_costs.ndim != 3:
        raise ValueError("Aggregated costs must be (H, W, D).")
    disp = np.argmin(aggregated_costs, axis=2).astype(np.float32)
    return disp


def run_sgm(cost_volume: np.ndarray, P1: float, P2: float, directions=None) -> np.ndarray:
    """
    Backward-compatible wrapper returning disparity via SGM + WTA.
    """
    aggregated = semi_global_matching(cost_volume, P1=P1, P2=P2, directions=directions)
    return winner_take_all(aggregated)

