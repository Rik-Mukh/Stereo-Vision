"""
Utility helpers for I/O and preprocessing.
"""

import cv2
import numpy as np


def load_image(path: str, color: bool = True) -> np.ndarray:
    """
    Load an image from disk. Color loads as BGR to match OpenCV defaults.
    """
    pass


def save_image(path: str, image: np.ndarray) -> None:
    """
    Save an image to disk.
    """
    pass


def resize_and_pad(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """
    Resize an image to fit within size (W, H) while preserving aspect ratio,
    then pad with zeros if necessary.
    """
    pass

