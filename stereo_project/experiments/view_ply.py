"""
Simple PLY point cloud viewer.

Usage:
    python -m stereo_project.experiments.view_ply <path_to_ply_file>
    python -m stereo_project.experiments.view_ply output.ply
"""

import argparse
import sys

from stereo_project.core import load_point_cloud_ply
from stereo_project.visualization import show_point_cloud


def main():
    parser = argparse.ArgumentParser(
        description="View a PLY point cloud file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("ply_file", help="Path to PLY file to view")
    args = parser.parse_args()
    
    try:
        print(f"Loading {args.ply_file}...")
        points = load_point_cloud_ply(args.ply_file)
        print(f"✓ Loaded {points.shape[0]:,} points")
        print(f"  X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
        print(f"  Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
        print(f"  Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
        print("\nOpening viewer... (close window when done)")
        show_point_cloud(points)
    except FileNotFoundError:
        print(f"✗ Error: File '{args.ply_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


