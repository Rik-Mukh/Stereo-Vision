"""
Rectification helpers to align stereo pairs into a common epipolar geometry.

Rectification enforces that corresponding points lie on the same image row,
which simplifies stereo matching to a 1D search along epipolar lines. OpenCV
is used only for remapping; the rectification math itself will be implemented
manually later.
"""

from typing import Tuple

import cv2
import numpy as np


def identity_rectification(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    For already-rectified datasets (e.g., Middlebury), return the inputs.
    """
    return left, right


def compute_rectification_maps(
    K_left: np.ndarray,
    K_right: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Compute rectification maps for left and right images given relative pose.

    Notes
    -----
    - Assume zero distortion.
    - Uses OpenCV's stereoRectify and initUndistortRectifyMap to obtain maps.
    """
    # OpenCV expects (width, height)
    w, h = image_size
    dist_zero = np.zeros(5, dtype=float)

    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
        K_left,
        dist_zero,
        K_right,
        dist_zero,
        (w, h),
        R,
        t,
        flags=cv2.CALIB_ZERO_DISPARITY,
    )

    map1_x, map1_y = cv2.initUndistortRectifyMap(
        K_left, dist_zero, R1, P1, (w, h), cv2.CV_32FC1
    )
    map2_x, map2_y = cv2.initUndistortRectifyMap(
        K_right, dist_zero, R2, P2, (w, h), cv2.CV_32FC1
    )

    return (map1_x, map1_y), (map2_x, map2_y)


def rectify_with_maps(
    left: np.ndarray,
    right: np.ndarray,
    map_left: tuple[np.ndarray, np.ndarray],
    map_right: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply rectification maps (xmap, ymap) to left and right images using cv2.remap.
    This aligns epipolar lines horizontally so the stereo matcher can search
    along image rows.
    """
    left_rect = cv2.remap(
        left, map_left[0], map_left[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    right_rect = cv2.remap(
        right, map_right[0], map_right[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    return left_rect, right_rect
