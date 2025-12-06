"""
Feature detection and matching using ORB and brute-force Hamming matching.
"""

from typing import Iterable, Tuple

import cv2
import numpy as np


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR images to grayscale when needed, and ensure uint8 format."""
    # Handle BGR images
    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Convert float32 [0, 1] to uint8 [0, 255] if needed
    if gray.dtype == np.float32 or gray.dtype == np.float64:
        if gray.max() <= 1.0:
            gray = (gray * 255.0).astype(np.uint8)
        else:
            gray = gray.astype(np.uint8)
    elif gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)
    
    return gray


def _draw_matches(
    img1: np.ndarray,
    img2: np.ndarray,
    kps1: Iterable[cv2.KeyPoint],
    kps2: Iterable[cv2.KeyPoint],
    matches: Iterable[cv2.DMatch],
    max_draw: int = 50,
) -> np.ndarray:
    """
    Simple debug helper to visualize a subset of matches.
    Converts images to color if they are grayscale.
    """
    subset = list(matches)[:max_draw]
    img1_vis = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if img1.ndim == 2 else img1.copy()
    img2_vis = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) if img2.ndim == 2 else img2.copy()
    return cv2.drawMatches(
        img1_vis,
        list(kps1),
        img2_vis,
        list(kps2),
        subset,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def detect_and_match_features(
    img1: np.ndarray,
    img2: np.ndarray,
    max_features: int = 8000,
    ratio_thresh: float = 0.75,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect and match features between two images using ORB and Lowe's ratio test.

    Parameters
    ----------
    img1, img2 : np.ndarray
        Grayscale images (H, W) or BGR images. If BGR, convert to grayscale internally.
    max_features : int
        Maximum number of features to detect in each image.
    ratio_thresh : float
        Lowe's ratio threshold for filtering matches.

    Returns
    -------
    pts1 : np.ndarray, shape (N, 2)
        Matched keypoints in image 1 (x, y).
    pts2 : np.ndarray, shape (N, 2)
        Corresponding keypoints in image 2 (x, y).
    """

    gray1 = _to_grayscale(img1)
    gray2 = _to_grayscale(img2)

    orb = cv2.ORB_create(nfeatures=max_features)
    kps1, desc1 = orb.detectAndCompute(gray1, None)
    kps2, desc2 = orb.detectAndCompute(gray2, None)

    if desc1 is None or desc2 is None or len(kps1) == 0 or len(kps2) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = matcher.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    if not good_matches:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    good_matches.sort(key=lambda m: m.distance)
    if len(good_matches) > max_features:
        good_matches = good_matches[:max_features]

    pts1 = np.array([kps1[m.queryIdx].pt for m in good_matches], dtype=np.float32)
    pts2 = np.array([kps2[m.trainIdx].pt for m in good_matches], dtype=np.float32)
    return pts1, pts2
