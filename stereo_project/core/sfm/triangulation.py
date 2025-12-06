"""
Linear triangulation and reprojection error utilities.
"""

import numpy as np


def triangulate_points(
    pts1: np.ndarray,
    pts2: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
) -> np.ndarray:
    """
    Linear triangulation of 3D points from two views.

    Parameters
    ----------
    pts1, pts2 : (N, 2)
        Corresponding pixel coordinates in image 1 and image 2.
    P1, P2 : (3, 4)
        Projection matrices for camera 1 and camera 2 in the SAME world frame.

    Returns
    -------
    X : (N, 3)
        Triangulated 3D points in Euclidean coordinates.

    Notes
    -----
    We stack four linear equations (A X = 0) derived from the cross-product
    form of the projection equation x × (P X) = 0, then solve via SVD.
    """
    if pts1.shape != pts2.shape:
        raise ValueError("pts1 and pts2 must have the same shape")
    if pts1.ndim != 2 or pts1.shape[1] != 2:
        raise ValueError("pts1 and pts2 must have shape (N, 2)")

    n = pts1.shape[0]
    X_out = np.zeros((n, 3), dtype=float)

    for i in range(n):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]

        A = np.array(
            [
                y1 * P1[2, :] - P1[1, :],
                x1 * P1[2, :] - P1[0, :],
                y2 * P2[2, :] - P2[1, :],
                x2 * P2[2, :] - P2[0, :],
            ],
            dtype=float,
        )

        _, _, vh = np.linalg.svd(A)
        X_h = vh[-1]
        if np.abs(X_h[3]) < 1e-12:
            X_out[i] = X_h[:3]
        else:
            X_out[i] = X_h[:3] / X_h[3]

    return X_out


def reprojection_errors(
    X: np.ndarray,
    pts: np.ndarray,
    P: np.ndarray,
) -> np.ndarray:
    """
    Compute reprojection errors for 3D points X onto image points pts using P.

    Parameters
    ----------
    X : (N, 3)
        3D points.
    pts : (N, 2)
        Observed 2D points.
    P : (3, 4)
        Projection matrix.

    Returns
    -------
    errors : (N,)
        Euclidean reprojection error per point in pixels.
    """
    if X.shape[0] != pts.shape[0]:
        raise ValueError("X and pts must have the same number of points")

    n = X.shape[0]
    errs = np.zeros(n, dtype=float)
    for i in range(n):
        X_h = np.array([X[i, 0], X[i, 1], X[i, 2], 1.0], dtype=float)
        proj = P @ X_h
        if np.abs(proj[2]) < 1e-12:
            errs[i] = np.inf
            continue
        proj_xy = proj[:2] / proj[2]
        errs[i] = np.linalg.norm(proj_xy - pts[i])
    return errs
