"""
Depth reconstruction and point cloud generation from disparity.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # avoid circular import at runtime
    from .camera import CameraIntrinsics


def disparity_to_depth(
    disparity: np.ndarray,
    fx: float,
    baseline: float,
    min_disparity: float = 0.1,
) -> np.ndarray:
    """
    Convert disparity to depth using Z = fx * baseline / disparity.
    """
    disp = disparity.astype(np.float32)
    disp = np.maximum(disp, min_disparity)  # clamp to avoid div by zero
    depth = (fx * baseline) / disp
    return depth


def depth_to_point_cloud(
    depth: np.ndarray,
    intrinsics: "CameraIntrinsics",
) -> np.ndarray:
    """
    Back-project depth map to 3D camera coordinates using pinhole model.

    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy
    Z = depth
    """
    H, W = depth.shape
    ys, xs = np.indices((H, W))
    Z = depth.astype(np.float32)
    valid = Z > 0
    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    Z = Z[valid]

    X = (xs - intrinsics.cx) * Z / intrinsics.fx
    Y = (ys - intrinsics.cy) * Z / intrinsics.fy
    points = np.stack([X, Y, Z], axis=1)
    return points


def save_point_cloud_ply(points: np.ndarray, path: str) -> None:
    """
    Save point cloud (N, 3) to ASCII PLY (geometry only).
    """
    points = points.astype(np.float32)
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {points.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        np.savetxt(f, points, fmt="%.6f %.6f %.6f")


def load_point_cloud_ply(path: str) -> np.ndarray:
    """
    Load point cloud from ASCII PLY file.
    Returns (N, 3) array of points.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Find end of header
    header_end = next(i for i, line in enumerate(lines) if "end_header" in line.lower())
    
    # Read points (skip header)
    points = []
    for line in lines[header_end + 1 :]:
        line = line.strip()
        if line:
            coords = line.split()
            if len(coords) >= 3:
                try:
                    points.append([float(coords[0]), float(coords[1]), float(coords[2])])
                except ValueError:
                    continue  # Skip invalid lines
    
    return np.array(points, dtype=np.float32)
