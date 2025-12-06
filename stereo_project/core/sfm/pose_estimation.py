"""
Relative pose recovery from an Essential matrix using cheirality.
"""

from typing import List, Tuple

import numpy as np

from .triangulation import triangulate_points


def decompose_essential(E: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Decompose Essential matrix E into the four possible (R, t) pairs.
    """
    U, _, Vt = np.linalg.svd(E)

    # Enforce proper singular values (1,1,0) to stabilize decomposition.
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)

    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    t = U[:, 2]
    candidates = [
        (R1, t),
        (R1, -t),
        (R2, t),
        (R2, -t),
    ]

    # Ensure rotations are proper (det ~ +1)
    valid = []
    for R, t_vec in candidates:
        if np.linalg.det(R) < 0:
            R = -R
            t_vec = -t_vec
        valid.append((R, t_vec))
    return valid


def _cheirality_count(R: np.ndarray, t: np.ndarray, pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray) -> int:
    """
    Count how many triangulated points lie in front of both cameras.
    """
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])

    X = triangulate_points(pts1, pts2, P1, P2)
    # Z in cam1 frame is X[:,2]; in cam2 frame is (R X + t) z-component
    X_cam2 = (R @ X.T).T + t.reshape(1, 3)
    in_front = (X[:, 2] > 0) & (X_cam2[:, 2] > 0)
    return int(np.sum(in_front))


def choose_correct_pose(
    candidates: List[Tuple[np.ndarray, np.ndarray]],
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select the physically valid (R, t) pair using the cheirality condition.
    """
    if pts1.shape[0] != pts2.shape[0]:
        raise ValueError("pts1 and pts2 must have the same number of points")
    if pts1.shape[0] < 1:
        raise ValueError("Need at least one correspondence to choose pose")

    # Optionally subsample to speed up.
    max_pts = 200
    if pts1.shape[0] > max_pts:
        idx = np.random.choice(pts1.shape[0], max_pts, replace=False)
        pts1_sub = pts1[idx]
        pts2_sub = pts2[idx]
    else:
        pts1_sub = pts1
        pts2_sub = pts2

    best_count = -1
    best_pose = candidates[0]
    for R, t in candidates:
        count = _cheirality_count(R, t, pts1_sub, pts2_sub, K)
        if count > best_count:
            best_count = count
            best_pose = (R, t)
    return best_pose


def recover_pose_from_essential(
    E: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full pipeline: decompose E, run cheirality test, and return the best (R, t).
    """
    candidates = decompose_essential(E)
    R, t = choose_correct_pose(candidates, pts1, pts2, K)
    return R, t

