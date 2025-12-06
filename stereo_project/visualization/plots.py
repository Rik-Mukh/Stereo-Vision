"""
Matplotlib-based plotting utilities.
"""

import matplotlib.pyplot as plt
import numpy as np


def show_disparity_map(disparity: np.ndarray, title: str = "Disparity") -> None:
    """
    Display a disparity map with normalization to [0, 1] and perceptual colormap.
    """
    disp = disparity.astype(np.float32)
    vmax = np.percentile(disp[disp > 0], 99) if np.any(disp > 0) else 1.0
    norm = np.clip(disp / (vmax + 1e-6), 0, 1)
    plt.figure()
    plt.imshow(norm, cmap="plasma")
    plt.title(title)
    plt.colorbar(label="normalized disparity")
    plt.axis("off")
    plt.show()


def show_depth_map(depth: np.ndarray, title: str = "Depth") -> None:
    """
    Display a depth map; inverse depth helps visualize far points.
    """
    depth = depth.astype(np.float32)
    inv_depth = np.where(depth > 0, 1.0 / depth, 0.0)
    vmax = np.percentile(inv_depth[inv_depth > 0], 99) if np.any(inv_depth > 0) else 1.0
    norm = np.clip(inv_depth / (vmax + 1e-6), 0, 1)
    plt.figure()
    plt.imshow(norm, cmap="magma")
    plt.title(title)
    plt.colorbar(label="normalized inverse depth")
    plt.axis("off")
    plt.show()


def plot_cost_volume_slice(cost_volume: np.ndarray, disparity: int = 0):
    """
    Visualize a single disparity slice of the cost volume.
    """
    if cost_volume.ndim != 3:
        raise ValueError("cost_volume must have shape (H, W, D)")
    if disparity < 0 or disparity >= cost_volume.shape[2]:
        raise ValueError("disparity index out of range.")
    slice_img = cost_volume[:, :, disparity]
    plt.figure()
    plt.imshow(slice_img, cmap="viridis")
    plt.title(f"Cost volume slice d={disparity}")
    plt.colorbar(label="cost")
    plt.axis("off")
    plt.show()
