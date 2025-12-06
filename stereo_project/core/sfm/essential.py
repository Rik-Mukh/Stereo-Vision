"""
Essential matrix estimation via normalized 8-point algorithm and RANSAC.
"""

from typing import Tuple

import numpy as np


def normalize_points(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize 2D points so that centroid is at origin and mean distance is sqrt(2).
    """
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("pts must have shape (N, 2)")
    centroid = pts.mean(axis=0)
    pts_centered = pts - centroid
    mean_dist = np.mean(np.linalg.norm(pts_centered, axis=1))
    scale = np.sqrt(2) / mean_dist if mean_dist > 1e-9 else 1.0
    T = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=float)])
    pts_norm_h = (T @ pts_h.T).T
    return pts_norm_h[:, :2], T


def eight_point_essential(pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Compute the Essential matrix using the normalized 8-point algorithm.
    """
    if pts1.shape != pts2.shape or pts1.shape[0] < 8:
        raise ValueError("Need at least 8 correspondences with matching shapes.")

    invK = np.linalg.inv(K)
    pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1), dtype=float)])
    pts2_h = np.hstack([pts2, np.ones((pts2.shape[0], 1), dtype=float)])
    x1_cam = (invK @ pts1_h.T).T
    x2_cam = (invK @ pts2_h.T).T
    x1_cam /= x1_cam[:, 2:3]
    x2_cam /= x2_cam[:, 2:3]

    x1_norm, T1 = normalize_points(x1_cam[:, :2])
    x2_norm, T2 = normalize_points(x2_cam[:, :2])

    x1_h = np.hstack([x1_norm, np.ones((x1_norm.shape[0], 1), dtype=float)])
    x2_h = np.hstack([x2_norm, np.ones((x2_norm.shape[0], 1), dtype=float)])

    A = np.zeros((x1_h.shape[0], 9), dtype=float)
    A[:, 0] = x2_h[:, 0] * x1_h[:, 0]
    A[:, 1] = x2_h[:, 0] * x1_h[:, 1]
    A[:, 2] = x2_h[:, 0]
    A[:, 3] = x2_h[:, 1] * x1_h[:, 0]
    A[:, 4] = x2_h[:, 1] * x1_h[:, 1]
    A[:, 5] = x2_h[:, 1]
    A[:, 6] = x1_h[:, 0]
    A[:, 7] = x1_h[:, 1]
    A[:, 8] = 1.0

    _, _, vh = np.linalg.svd(A)
    E_tilde = vh[-1].reshape(3, 3)

    U, S, Vt = np.linalg.svd(E_tilde)
    S[2] = 0.0  # enforce rank-2
    E_rank2 = U @ np.diag(S) @ Vt

    E = T2.T @ E_rank2 @ T1
    norm = np.linalg.norm(E)
    return E / norm if norm > 0 else E


def _symmetric_epipolar_distance(E: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    Symmetric epipolar distance in normalized camera coordinates.
    """
    Ex1 = E @ x1.T
    Etx2 = E.T @ x2.T
    x2tEx1 = np.sum(x2 * (Ex1.T), axis=1)
    denom = Ex1[0] ** 2 + Ex1[1] ** 2 + Etx2[0] ** 2 + Etx2[1] ** 2
    denom = np.where(denom < 1e-12, 1e-12, denom)
    return (x2tEx1**2) / denom


def estimate_essential_matrix_ransac(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    num_iters: int = 2000,
    threshold: float = 0.001,
    confidence: float = 0.99,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate Essential matrix with a RANSAC loop around the 8-point algorithm.

    Epipolar constraint: x2^T E x1 = 0 for corresponding normalized points.
    RANSAC samples minimal sets, fits E, then counts inliers via symmetric
    epipolar distance to reject outliers.
    """
    if pts1.shape != pts2.shape or pts1.shape[0] < 8:
        raise ValueError("Need at least 8 correspondences with matching shapes.")

    pts1 = np.asarray(pts1, dtype=float)
    pts2 = np.asarray(pts2, dtype=float)
    n_pts = pts1.shape[0]

    invK = np.linalg.inv(K)
    pts1_h = np.hstack([pts1, np.ones((n_pts, 1), dtype=float)])
    pts2_h = np.hstack([pts2, np.ones((n_pts, 1), dtype=float)])
    x1_cam = (invK @ pts1_h.T).T
    x2_cam = (invK @ pts2_h.T).T
    x1_cam /= x1_cam[:, 2:3]
    x2_cam /= x2_cam[:, 2:3]

    best_E = None
    best_inliers = np.zeros(n_pts, dtype=bool)
    best_count = 0
    i = 0
    max_iters = num_iters
    minimal_size = 8
    rng = np.random.default_rng()

    while i < max_iters:
        sample_idx = rng.choice(n_pts, minimal_size, replace=False)
        try:
            E_candidate = eight_point_essential(pts1[sample_idx], pts2[sample_idx], K)
        except np.linalg.LinAlgError:
            i += 1
            continue

        d = _symmetric_epipolar_distance(E_candidate, x1_cam, x2_cam)
        inliers_mask = d < threshold
        inlier_count = int(np.sum(inliers_mask))

        if inlier_count > best_count:
            best_count = inlier_count
            best_inliers = inliers_mask
            best_E = E_candidate

            inlier_ratio = inlier_count / n_pts
            if inlier_ratio > 0:
                # update max_iters to reach desired confidence
                prob_no_outliers = 1 - inlier_ratio**minimal_size
                prob_no_outliers = np.clip(prob_no_outliers, 1e-9, 1 - 1e-9)
                max_iters = min(
                    max_iters,
                    int(np.log(1 - confidence) / np.log(prob_no_outliers)) + 1,
                )
        i += 1

    if best_E is None or best_count < minimal_size:
        return (
            np.zeros((3, 3), dtype=float),
            np.empty((0, 2), dtype=float),
            np.empty((0, 2), dtype=float),
        )

    inlier_pts1 = pts1[best_inliers]
    inlier_pts2 = pts2[best_inliers]
    if inlier_pts1.shape[0] >= minimal_size:
        try:
            best_E = eight_point_essential(inlier_pts1, inlier_pts2, K)
        except np.linalg.LinAlgError:
            pass

    return best_E, inlier_pts1, inlier_pts2


def estimate_essential_matrix(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    num_iters: int = 2000,
    threshold: float = 0.001,
    confidence: float = 0.99,
) -> np.ndarray:
    """
    Convenience wrapper returning only E (kept for compatibility).
    """
    E, _, _ = estimate_essential_matrix_ransac(pts1, pts2, K, num_iters, threshold, confidence)
    return E

