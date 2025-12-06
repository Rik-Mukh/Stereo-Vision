"""
Camera and calibration utilities for stereo setups.

Only uses OpenCV for image I/O and basic operations; heavy lifting for
rectification and stereo math is implemented manually elsewhere.
"""

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class CameraIntrinsics:
    """
    Intrinsic parameters for a single camera.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0

    def matrix(self) -> np.ndarray:
        """
        Return the 3x3 intrinsic calibration matrix.
        """
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass
class StereoRig:
    """
    Stereo rig description with left/right intrinsics and baseline.
    """

    left: CameraIntrinsics
    right: CameraIntrinsics
    baseline: float  # meters
    doffs: float = 0.0


def _parse_matrix(value: str) -> np.ndarray:
    """
    Parse a matrix encoded as cam0=[a,b,c;d,e,f;g,h,i].
    """
    inner = value.strip()
    if inner.startswith("[") and inner.endswith("]"):
        inner = inner[1:-1]
    rows = inner.split(";")
    data = []
    for row in rows:
        cleaned = row.replace(",", " ")
        numbers = [float(tok) for tok in cleaned.split() if tok]
        if numbers:
            data.append(numbers)
    mat = np.array(data, dtype=np.float64)
    if mat.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix, got shape {mat.shape}")
    return mat


def load_middlebury_calibration(calib_path: str) -> StereoRig:
    """
    Parse a Middlebury calib.txt and return a StereoRig.

    Assumptions
    -----------
    - cam0 and cam1 are given as 3x3 matrices in the file.
    - baseline is in millimeters; converted to meters.
    - doffs may be provided (default 0).
    - width, height are provided.
    """
    cam0 = cam1 = None
    baseline_mm = None
    doffs = 0.0
    width = height = None

    with open(calib_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if key == "cam0":
                cam0 = _parse_matrix(value)
            elif key == "cam1":
                cam1 = _parse_matrix(value)
            elif key == "baseline":
                baseline_mm = float(value)
            elif key == "doffs":
                doffs = float(value)
            elif key == "width":
                width = int(float(value))
            elif key == "height":
                height = int(float(value))

    if cam0 is None or cam1 is None:
        raise ValueError("Missing cam0 or cam1 in calibration file.")
    if baseline_mm is None:
        raise ValueError("Missing baseline in calibration file.")
    if width is None or height is None:
        raise ValueError("Missing image dimensions in calibration file.")

    baseline_m = baseline_mm * 1e-3  # Middlebury baseline is given in mm.

    left_intr = CameraIntrinsics(
        fx=cam0[0, 0],
        fy=cam0[1, 1],
        cx=cam0[0, 2],
        cy=cam0[1, 2],
        width=width,
        height=height,
    )
    right_intr = CameraIntrinsics(
        fx=cam1[0, 0],
        fy=cam1[1, 1],
        cx=cam1[0, 2],
        cy=cam1[1, 2],
        width=width,
        height=height,
    )

    return StereoRig(left=left_intr, right=right_intr, baseline=baseline_m, doffs=doffs)


def read_stereo_pair(left_image_path: str, right_image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read left and right images in BGR, convert to grayscale float32 in [0, 1].
    """
    left_bgr = cv2.imread(left_image_path, cv2.IMREAD_COLOR)
    right_bgr = cv2.imread(right_image_path, cv2.IMREAD_COLOR)
    if left_bgr is None:
        raise FileNotFoundError(f"Could not read left image at {left_image_path}")
    if right_bgr is None:
        raise FileNotFoundError(f"Could not read right image at {right_image_path}")

    left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return left_gray, right_gray


def load_calibration(calib_path: str) -> StereoRig:
    """
    Load calibration parameters from a Middlebury-style calib.txt.
    """
    return load_middlebury_calibration(calib_path)


def create_rectification_maps(calibration: dict):
    """
    Prepare rectification/remapping transforms from loaded calibration.

    Notes
    -----
    Actual rectification math is handled in rectification.py; this helper
    should format and validate calibration data before rectification.
    """
    pass
