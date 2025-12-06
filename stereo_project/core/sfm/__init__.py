"""
Structure-from-Motion (SfM) utilities.

This subpackage will later include:
- Feature detection and matching (SIFT/ORB-style)
- Essential matrix estimation
- Pose (R, t) recovery
- Sparse triangulation
"""

from .features import detect_and_match_features
from .essential import estimate_essential_matrix
from .pose_estimation import recover_pose_from_essential
from .triangulation import triangulate_points

__all__ = [
    "detect_and_match_features",
    "estimate_essential_matrix",
    "recover_pose_from_essential",
    "triangulate_points",
]

