"""
Demo: two-view SfM + stereo depth on arbitrary image pair with known intrinsics.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from stereo_project.core.camera import CameraIntrinsics
from stereo_project.core.cost_volume import (
    compute_census_cost_volume,
    compute_sad_cost_volume,
    normalize_cost_volume,
)
from stereo_project.core.depth import depth_to_point_cloud, disparity_to_depth, save_point_cloud_ply
from stereo_project.core.disparity import (
    fill_invalid_disparities,
    left_right_consistency_check,
    speckle_filter,
    subpixel_refine_disparity,
)
from stereo_project.core.rectification import compute_rectification_maps, rectify_with_maps
from stereo_project.core.sfm.essential import estimate_essential_matrix_ransac
from stereo_project.core.sfm.features import detect_and_match_features
from stereo_project.core.sfm.pose_estimation import recover_pose_from_essential
from stereo_project.core.sgm import compute_adaptive_P2, semi_global_matching, winner_take_all
from stereo_project.visualization.plots import show_depth_map, show_disparity_map


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run two-view SfM + stereo pipeline on a custom pair.")
    parser.add_argument("--left", required=True, help="Path to left image.")
    parser.add_argument("--right", required=True, help="Path to right image.")
    parser.add_argument("--fx", type=float, default=None, help="Focal length fx (pixels).")
    parser.add_argument("--fy", type=float, default=None, help="Focal length fy (pixels).")
    parser.add_argument("--cx", type=float, default=None, help="Principal point cx (pixels).")
    parser.add_argument("--cy", type=float, default=None, help="Principal point cy (pixels).")
    parser.add_argument("--intrinsics", type=str, default=None, help="Optional path to txt with fx fy cx cy (space/line separated).")
    parser.add_argument("--max_disparity", type=int, default=256, help="Maximum disparity for stereo.")
    parser.add_argument("--method", choices=["sad", "census"], default="census", help="Stereo matching cost type.")
    parser.add_argument("--downscale", type=float, default=1.0, help="Downscale factor for speed (e.g., 0.5).")
    parser.add_argument("--show_pointcloud", action="store_true", help="Print basic point cloud stats (no viewer).")
    parser.add_argument("--normalize_cost", action="store_true", help="Per-pixel min-max normalize cost volumes.")
    parser.add_argument("--lr_threshold", type=float, default=1.5, help="Left-right consistency threshold.")
    parser.add_argument("--speckle_size", type=int, default=100, help="Max speckle size to remove.")
    parser.add_argument("--census_window", type=int, default=7, help="Census window size (odd).")
    parser.add_argument("--save_ply", type=str, default=None, help="Optional path to save point cloud as PLY.")
    parser.add_argument("--debug", action="store_true", help="Verbose debug logging for rectification and disparity.")
    return parser.parse_args()


def _load_intrinsics(path: str) -> tuple[float, float, float, float]:
    vals = np.loadtxt(path).flatten()
    if vals.size < 4:
        raise ValueError("Intrinsics file must contain at least fx fy cx cy.")
    return float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])


def _read_images(left_path: str, right_path: str) -> tuple[np.ndarray, np.ndarray]:
    left_bgr = cv2.imread(left_path, cv2.IMREAD_COLOR)
    right_bgr = cv2.imread(right_path, cv2.IMREAD_COLOR)
    if left_bgr is None:
        raise FileNotFoundError(f"Could not read left image at {left_path}")
    if right_bgr is None:
        raise FileNotFoundError(f"Could not read right image at {right_path}")
    left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return left_gray, right_gray


def main():
    args = _parse_args()

    def dbg(msg: str) -> None:
        if args.debug:
            print(f"[debug] {msg}")

    if args.intrinsics:
        fx, fy, cx, cy = _load_intrinsics(args.intrinsics)
    else:
        if None in (args.fx, args.fy, args.cx, args.cy):
            raise ValueError("Provide fx, fy, cx, cy or --intrinsics file.")
        fx, fy, cx, cy = args.fx, args.fy, args.cx, args.cy

    left_gray, right_gray = _read_images(args.left, args.right)
    dbg(f"Input left stats: min={left_gray.min():.4f}, max={left_gray.max():.4f}, mean={left_gray.mean():.4f}")
    dbg(f"Input right stats: min={right_gray.min():.4f}, max={right_gray.max():.4f}, mean={right_gray.mean():.4f}")
    h, w = left_gray.shape[:2]

    if args.downscale != 1.0:
        scale = float(args.downscale)
        new_size = (int(w * scale), int(h * scale))
        left_gray = cv2.resize(left_gray, new_size, interpolation=cv2.INTER_AREA)
        right_gray = cv2.resize(right_gray, new_size, interpolation=cv2.INTER_AREA)
        fx *= scale
        fy *= scale
        cx *= scale
        cy *= scale
        h, w = left_gray.shape[:2]
        print(f"[info] Downscaled to {w}x{h}, updated intrinsics.")
        dbg(f"After downscale left stats: min={left_gray.min():.4f}, max={left_gray.max():.4f}, mean={left_gray.mean():.4f}")
        dbg(f"After downscale right stats: min={right_gray.min():.4f}, max={right_gray.max():.4f}, mean={right_gray.mean():.4f}")

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)

    # Feature detection and matching (initial)
    t0 = time.perf_counter()
    pts1_init, pts2_init = detect_and_match_features(left_gray, right_gray, max_features=8000, ratio_thresh=0.8)
    t1 = time.perf_counter()
    print(f"[info] Raw matches (initial): {len(pts1_init)} (time {t1 - t0:.3f}s)")
    dbg(f"First 5 matches pts1: {pts1_init[:5]} pts2: {pts2_init[:5]}")
    if len(pts1_init) < 8:
        raise RuntimeError("Not enough matches to estimate homography/Essential matrix.")

    # Pre-align right image to left using homography to reduce rotation
    H, mask_H = cv2.findHomography(pts2_init, pts1_init, cv2.RANSAC, 4.0)
    if H is not None:
        right_gray_aligned = cv2.warpPerspective(right_gray, H, (w, h))
        dbg(f"Homography inliers: {int(mask_H.sum()) if mask_H is not None else 0} / {len(mask_H) if mask_H is not None else 0}")
        dbg(f"H:\n{H}")
    else:
        right_gray_aligned = right_gray
        dbg("Homography failed; using original right image")

    # Re-detect matches on the aligned pair for Essential matrix
    pts1, pts2 = detect_and_match_features(left_gray, right_gray_aligned, max_features=8000, ratio_thresh=0.8)
    t1b = time.perf_counter()
    print(f"[info] Raw matches (post-align): {len(pts1)} (time {t1b - t1:.3f}s)")
    dbg(f"First 5 matches (post-align) pts1: {pts1[:5]} pts2: {pts2[:5]}")
    if len(pts1) < 8:
        raise RuntimeError("Not enough matches after homography alignment for Essential matrix.")

    # Essential matrix estimation with RANSAC
    E, inliers1, inliers2 = estimate_essential_matrix_ransac(pts1, pts2, K, threshold=0.003)
    t2 = time.perf_counter()
    print(f"[info] Inliers after RANSAC: {len(inliers1)} / {len(pts1)} (time {t2 - t1:.3f}s)")
    dbg(f"Inlier sample pts1: {inliers1[:5]} pts2: {inliers2[:5]}")
    # Analyze E
    try:
        Ue, Se, Vte = np.linalg.svd(E)
        dbg(f"E singular values: {Se}, det(U)={np.linalg.det(Ue):.4f}, det(Vt)={np.linalg.det(Vte):.4f}")
    except Exception as exc:
        dbg(f"Failed SVD on E: {exc}")

    # Pose recovery with cheirality
    R, t = recover_pose_from_essential(E, inliers1, inliers2, K)
    t3 = time.perf_counter()
    baseline = float(np.linalg.norm(t))
    if baseline <= 0:
        raise RuntimeError("Recovered translation has zero length; cannot compute baseline.")
    print(f"[info] Pose recovered (baseline {baseline:.4f} m) (time {t3 - t2:.3f}s)")
    # Cheirality counts for all candidates (debug)
    if args.debug:
        from stereo_project.core.sfm.pose_estimation import decompose_essential, _cheirality_count

        candidates = decompose_essential(E)
        counts = []
        for i, (Rc, tc) in enumerate(candidates):
            cnt = _cheirality_count(Rc, tc, inliers1, inliers2, K)
            counts.append(cnt)
        dbg(f"Cheirality counts per candidate: {counts} (selected max={max(counts) if counts else 0})")
        dbg(f"R:\n{R}\nt: {t}")

    # Rectification maps
    map_left, map_right = compute_rectification_maps(K, K, R, t, (w, h))
    left_rect, right_rect = rectify_with_maps(left_gray, right_gray, map_left, map_right)
    t4 = time.perf_counter()
    print(f"[info] Rectification done (time {t4 - t3:.3f}s)")
    dbg(f"Rectified left stats: min={left_rect.min():.4f}, max={left_rect.max():.4f}, mean={left_rect.mean():.4f}")
    dbg(f"Rectified right stats: min={right_rect.min():.4f}, max={right_rect.max():.4f}, mean={right_rect.mean():.4f}")

    # Stereo matching
    if args.method == "sad":
        cost_vol = compute_sad_cost_volume(left_rect, right_rect, args.max_disparity)
        cost_vol_r = compute_sad_cost_volume(right_rect, left_rect, args.max_disparity, right_to_left=True)
    else:
        cost_vol = compute_census_cost_volume(left_rect, right_rect, args.max_disparity, window_size=args.census_window)
        cost_vol_r = compute_census_cost_volume(
            right_rect, left_rect, args.max_disparity, window_size=args.census_window, right_to_left=True
        )

    if args.normalize_cost:
        cost_vol = normalize_cost_volume(cost_vol, method="minmax")
        cost_vol_r = normalize_cost_volume(cost_vol_r, method="minmax")
    dbg(f"Cost volume stats: min={cost_vol.min():.4f}, max={cost_vol.max():.4f}, mean={cost_vol.mean():.4f}")

    P2_adaptive = compute_adaptive_P2(left_rect)
    agg = semi_global_matching(cost_vol, P1=0.8, P2=P2_adaptive)
    disp_int = winner_take_all(agg)
    disp_sub = subpixel_refine_disparity(agg, disp_int)
    dbg(f"Disp int stats: min={disp_int.min():.3f}, max={disp_int.max():.3f}, mean={disp_int.mean():.3f}, nonzero={(disp_int>0).sum()}")
    dbg(f"Disp sub stats: min={disp_sub.min():.3f}, max={disp_sub.max():.3f}, mean={disp_sub.mean():.3f}, nonzero={(disp_sub>0).sum()}")

    # Right disparity for LR check
    agg_r = semi_global_matching(cost_vol_r, P1=0.8, P2=compute_adaptive_P2(right_rect))
    disp_r = winner_take_all(agg_r)
    lr_mask = left_right_consistency_check(disp_sub, disp_r, max_diff=args.lr_threshold)
    disp_sub[~lr_mask] = 0.0

    disp_filtered = speckle_filter(disp_sub, valid_mask=lr_mask, max_speckle_size=args.speckle_size)
    disp_filled = fill_invalid_disparities(disp_filtered, invalid_value=0.0)
    dbg(f"LR mask kept {(lr_mask).sum()} pixels of {lr_mask.size}")
    dbg(f"Disp after LR/speckle/fill: min={disp_filled.min():.3f}, max={disp_filled.max():.3f}, mean={disp_filled.mean():.3f}, nonzero={(disp_filled>0).sum()}")
    t5 = time.perf_counter()
    print(f"[info] Stereo matching + refinement done (time {t5 - t4:.3f}s)")

    # Depth and point cloud
    depth = disparity_to_depth(disp_filled, fx=fx, baseline=baseline)
    intr = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=w, height=h)
    points = depth_to_point_cloud(depth, intr)
    print(f"[info] Depth computed. Valid 3D points: {points.shape[0]}")
    dbg(f"Depth stats: min={depth.min():.3f}, max={depth.max():.3f}, mean={depth.mean():.3f}")
    if args.show_pointcloud:
        print(f"[info] Point cloud stats: min {points.min(axis=0)}, max {points.max(axis=0)}")
    if args.save_ply:
        save_point_cloud_ply(points, args.save_ply)
        print(f"[info] Saved point cloud to {args.save_ply}")

    # Visualizations
    stacked = np.concatenate([left_rect, right_rect], axis=0)
    cv2.imshow("Rectified (stacked vertically)", stacked)
    cv2.waitKey(1)
    show_disparity_map(disp_filled, title="Disparity (refined)")
    show_depth_map(depth, title="Depth (inverse depth colormap)")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

