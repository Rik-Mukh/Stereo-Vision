"""Core stereo vision algorithms and utilities."""

from .camera import (
    CameraIntrinsics,
    StereoRig,
    load_calibration,
    create_rectification_maps,
    load_middlebury_calibration,
    read_stereo_pair,
)
from .rectification import (
    identity_rectification,
    compute_rectification_maps,
    rectify_with_maps,
)
from .cost_volume import (
    compute_sad_cost_volume,
    census_transform,
    compute_census_cost_volume,
    normalize_cost_volume,
    compute_confidence,
)
from .sgm import semi_global_matching, winner_take_all, run_sgm, compute_adaptive_P2
from .disparity import (
    subpixel_refine_disparity,
    left_right_consistency_check,
    confidence_filter,
    speckle_filter,
    fill_invalid_disparities,
    fill_invalid_bilateral,
    median_filter_disparity,
    weighted_median_filter,
    bilateral_filter_disparity,
    subpixel_refine,  # legacy alias
)
from .depth import disparity_to_depth, depth_to_point_cloud, save_point_cloud_ply, load_point_cloud_ply
from .utils import load_image, save_image, resize_and_pad
from . import sfm

__all__ = [
    "CameraIntrinsics",
    "StereoRig",
    "load_calibration",
    "create_rectification_maps",
    "load_middlebury_calibration",
    "read_stereo_pair",
    "identity_rectification",
    "compute_rectification_maps",
    "rectify_with_maps",
    "compute_sad_cost_volume",
    "census_transform",
    "compute_census_cost_volume",
    "normalize_cost_volume",
    "compute_confidence",
    "semi_global_matching",
    "winner_take_all",
    "run_sgm",
    "compute_adaptive_P2",
    "subpixel_refine_disparity",
    "left_right_consistency_check",
    "confidence_filter",
    "speckle_filter",
    "fill_invalid_disparities",
    "fill_invalid_bilateral",
    "median_filter_disparity",
    "weighted_median_filter",
    "bilateral_filter_disparity",
    "subpixel_refine",
    "disparity_to_depth",
    "depth_to_point_cloud",
    "save_point_cloud_ply",
    "load_point_cloud_ply",
    "load_image",
    "save_image",
    "resize_and_pad",
    "sfm",
]