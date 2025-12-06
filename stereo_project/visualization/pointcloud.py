"""
Point cloud visualization utilities (optionally using Open3D).
"""

import numpy as np


def show_point_cloud(points: np.ndarray) -> None:
    """
    Visualize a 3D point cloud using open3d if available, else matplotlib 3D.
    """
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        o3d.visualization.draw_geometries([pcd])
        return
    except Exception as exc:  # open3d missing or other failure
        print(f"open3d unavailable or failed ({exc}); falling back to matplotlib.")

    # Fallback to matplotlib 3D scatter.
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed for 3D projection

        pts = points.astype(np.float32)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.5, c=pts[:, 2], cmap="viridis")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()
    except Exception as exc:
        print(f"Matplotlib 3D fallback failed: {exc}")
