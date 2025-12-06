"""
End-to-end Middlebury stereo pipeline.
"""

import argparse
import time

import cv2
import numpy as np

from stereo_project.core import (
    load_middlebury_calibration,
    read_stereo_pair,
    identity_rectification,
    compute_sad_cost_volume,
    compute_census_cost_volume,
    normalize_cost_volume,
    compute_confidence,
    semi_global_matching,
    compute_adaptive_P2,
    winner_take_all,
    subpixel_refine_disparity,
    left_right_consistency_check,
    confidence_filter,
    speckle_filter,
    fill_invalid_disparities,
    fill_invalid_bilateral,
    median_filter_disparity,
    weighted_median_filter,
    bilateral_filter_disparity,
    disparity_to_depth,
    depth_to_point_cloud,
    save_point_cloud_ply,
)
from stereo_project.visualization import show_disparity_map, show_depth_map


def _adjust_intrinsics_for_scale(intrinsics, scale: float):
    """Scale intrinsics when images are downsampled."""
    intrinsics.fx *= scale
    intrinsics.fy *= scale
    intrinsics.cx *= scale
    intrinsics.cy *= scale
    intrinsics.width = int(round(intrinsics.width * scale))
    intrinsics.height = int(round(intrinsics.height * scale))
    return intrinsics


def run_pipeline(args):
    print("=" * 60)
    print("DEBUG: Starting pipeline")
    print("=" * 60)
    
    # Load calibration and images.
    rig = load_middlebury_calibration(args.calib)
    print(f"\nDEBUG: Calibration loaded")
    print(f"  Baseline: {rig.baseline} m")
    print(f"  Left intrinsics: fx={rig.left.fx:.2f}, fy={rig.left.fy:.2f}, cx={rig.left.cx:.2f}, cy={rig.left.cy:.2f}")
    print(f"  Left image size: {rig.left.width}x{rig.left.height}")
    print(f"  Right intrinsics: fx={rig.right.fx:.2f}, fy={rig.right.fy:.2f}, cx={rig.right.cx:.2f}, cy={rig.right.cy:.2f}")
    print(f"  Right image size: {rig.right.width}x{rig.right.height}")
    
    left_img, right_img = read_stereo_pair(args.left, args.right)
    print(f"\nDEBUG: Images loaded")
    print(f"  Left image: shape={left_img.shape}, dtype={left_img.dtype}, min={left_img.min():.4f}, max={left_img.max():.4f}, mean={left_img.mean():.4f}, std={left_img.std():.4f}")
    print(f"  Right image: shape={right_img.shape}, dtype={right_img.dtype}, min={right_img.min():.4f}, max={right_img.max():.4f}, mean={right_img.mean():.4f}, std={right_img.std():.4f}")

    if args.downscale != 1.0:
        new_size = (int(left_img.shape[1] * args.downscale), int(left_img.shape[0] * args.downscale))
        left_img = cv2.resize(left_img, new_size, interpolation=cv2.INTER_AREA)
        right_img = cv2.resize(right_img, new_size, interpolation=cv2.INTER_AREA)
        rig.left = _adjust_intrinsics_for_scale(rig.left, args.downscale)
        rig.right = _adjust_intrinsics_for_scale(rig.right, args.downscale)
        print(f"\nDEBUG: Images downscaled to {new_size}")
        print(f"  Left image after resize: shape={left_img.shape}, min={left_img.min():.4f}, max={left_img.max():.4f}")

    # Middlebury images are already rectified.
    left_rect, right_rect = identity_rectification(left_img, right_img)
    print(f"\nDEBUG: After rectification (identity)")
    print(f"  Left rect: shape={left_rect.shape}, min={left_rect.min():.4f}, max={left_rect.max():.4f}")
    print(f"  Right rect: shape={right_rect.shape}, min={right_rect.min():.4f}, max={right_rect.max():.4f}")

    # Build cost volume with configurable Census window size.
    census_window = args.census_window
    t0 = time.perf_counter()
    if args.method == "sad":
        cost_volume = compute_sad_cost_volume(left_rect, right_rect, args.max_disparity)
    else:
        cost_volume = compute_census_cost_volume(left_rect, right_rect, args.max_disparity, window_size=census_window)
    t1 = time.perf_counter()
    print(f"\nCost volume ({args.method}, window={census_window}) built in {t1 - t0:.2f}s")
    print(f"DEBUG: Cost volume statistics")
    print(f"  Shape: {cost_volume.shape}")
    print(f"  Dtype: {cost_volume.dtype}")
    print(f"  Min: {cost_volume.min():.4f}, Max: {cost_volume.max():.4f}, Mean: {cost_volume.mean():.4f}, Std: {cost_volume.std():.4f}")
    # Check a few sample pixels
    h, w = cost_volume.shape[:2]
    sample_y, sample_x = h // 2, w // 2
    print(f"  Sample pixel at ({sample_y}, {sample_x}):")
    print(f"    Cost range across disparities: min={cost_volume[sample_y, sample_x, :].min():.4f}, max={cost_volume[sample_y, sample_x, :].max():.4f}")
    print(f"    Best disparity (argmin): {cost_volume[sample_y, sample_x, :].argmin()}")
    print(f"    Cost at disparity 0: {cost_volume[sample_y, sample_x, 0]:.4f}")
    print(f"    Cost at disparity {args.max_disparity//2}: {cost_volume[sample_y, sample_x, args.max_disparity//2]:.4f}")
    print(f"    Cost at disparity {args.max_disparity-1}: {cost_volume[sample_y, sample_x, args.max_disparity-1]:.4f}")

    # Normalize cost volume for better discrimination
    if args.normalize_cost:
        cost_volume = normalize_cost_volume(cost_volume, method="minmax")
        print(f"\nDEBUG: Cost volume normalized (minmax)")
        print(f"  Min: {cost_volume.min():.4f}, Max: {cost_volume.max():.4f}, Mean: {cost_volume.mean():.4f}")
    
    # Compute confidence from cost volume (before SGM, on raw/normalized costs)
    confidence = compute_confidence(cost_volume, method="peak_ratio")
    print(f"\nDEBUG: Confidence computed")
    print(f"  Min: {confidence.min():.4f}, Max: {confidence.max():.4f}, Mean: {confidence.mean():.4f}")
    print(f"  Fraction with confidence > 0.3: {(confidence > 0.3).sum() / confidence.size:.4f}")
    print(f"  Fraction with confidence > 0.5: {(confidence > 0.5).sum() / confidence.size:.4f}")

    # Semi-Global Matching with adaptive P2.
    if args.adaptive_p2:
        P2 = compute_adaptive_P2(left_rect, P2_base=args.p2, P2_min=2.0, gradient_scale=10.0)
        print(f"\nDEBUG: Adaptive P2 computed: min={P2.min():.2f}, max={P2.max():.2f}, mean={P2.mean():.2f}")
    else:
        P2 = args.p2
    aggregated = semi_global_matching(cost_volume, P1=args.p1, P2=P2)
    t2 = time.perf_counter()
    print(f"\nSGM aggregated in {t2 - t1:.2f}s")
    print(f"DEBUG: Aggregated costs statistics")
    print(f"  Shape: {aggregated.shape}")
    print(f"  Dtype: {aggregated.dtype}")
    print(f"  Min: {aggregated.min():.4f}, Max: {aggregated.max():.4f}, Mean: {aggregated.mean():.4f}, Std: {aggregated.std():.4f}")
    print(f"  Sample pixel at ({sample_y}, {sample_x}):")
    print(f"    Aggregated cost range: min={aggregated[sample_y, sample_x, :].min():.4f}, max={aggregated[sample_y, sample_x, :].max():.4f}")
    print(f"    Best disparity (argmin): {aggregated[sample_y, sample_x, :].argmin()}")

    # Disparity estimation.
    disp_int = winner_take_all(aggregated)
    print(f"\nDEBUG: Integer disparity (before refinement)")
    print(f"  Shape: {disp_int.shape}, Dtype: {disp_int.dtype}")
    print(f"  Min: {disp_int.min():.4f}, Max: {disp_int.max():.4f}, Mean: {disp_int.mean():.4f}, Std: {disp_int.std():.4f}")
    print(f"  Unique values (first 20): {np.unique(disp_int)[:20]}")
    print(f"  Sample pixel at ({sample_y}, {sample_x}): disparity = {disp_int[sample_y, sample_x]:.4f}")
    
    disp = subpixel_refine_disparity(aggregated, disp_int)
    t3 = time.perf_counter()
    print(f"\nRefinement completed in {t3 - t2:.2f}s")
    print(f"DEBUG: Refined disparity")
    print(f"  Shape: {disp.shape}, Dtype: {disp.dtype}")
    print(f"  Min: {disp.min():.4f}, Max: {disp.max():.4f}, Mean: {disp.mean():.4f}, Std: {disp.std():.4f}")
    print(f"  Sample pixel at ({sample_y}, {sample_x}): disparity = {disp[sample_y, sample_x]:.4f}")
    print(f"  Fraction of pixels with disparity == 0: {(disp == 0).sum() / disp.size:.4f}")
    print(f"  Fraction of pixels with disparity == {args.max_disparity-1}: {(disp == args.max_disparity-1).sum() / disp.size:.4f}")

    # Optional post-processing with right disparity and filtering.
    if args.postprocess:
        print(f"\nDEBUG: Starting post-processing")
        # Right disparity (right-to-left matching).
        # Use right_to_left=True to correctly shift the left image LEFT by d.
        if args.method == "sad":
            cost_volume_r = compute_sad_cost_volume(left_rect, right_rect, args.max_disparity, right_to_left=True)
        else:
            cost_volume_r = compute_census_cost_volume(left_rect, right_rect, args.max_disparity, window_size=census_window, right_to_left=True)
        
        # Use same P2 (adaptive or scalar) for right disparity
        if args.adaptive_p2:
            P2_r = compute_adaptive_P2(right_rect, P2_base=args.p2, P2_min=2.0, gradient_scale=10.0)
        else:
            P2_r = args.p2
        aggregated_r = semi_global_matching(cost_volume_r, P1=args.p1, P2=P2_r)
        disp_r_int = winner_take_all(aggregated_r)
        disp_r = subpixel_refine_disparity(aggregated_r, disp_r_int)
        print(f"  Right disparity: min={disp_r.min():.4f}, max={disp_r.max():.4f}, mean={disp_r.mean():.4f}")
        
        # Debug: Check a sample pixel for LR consistency
        h, w = disp.shape
        sample_y, sample_x = h // 2, w // 2
        d_left = disp[sample_y, sample_x]
        x_right = int(round(sample_x - d_left))
        if 0 <= x_right < w:
            d_right_at_xr = disp_r[sample_y, x_right]
            print(f"  DEBUG LR consistency at sample pixel ({sample_y}, {sample_x}):")
            print(f"    Left disparity d_L = {d_left:.2f}")
            print(f"    Right image x-coord = {sample_x} - {d_left:.2f} = {x_right}")
            print(f"    Right disparity d_R at ({sample_y}, {x_right}) = {d_right_at_xr:.2f}")
            print(f"    Difference |d_L - d_R| = {abs(d_left - d_right_at_xr):.2f}")
            print(f"    Should pass if <= 1.0: {abs(d_left - d_right_at_xr) <= 1.0}")

        # Confidence-based filtering (before LR check)
        if args.confidence_threshold > 0:
            disp = confidence_filter(disp, confidence, threshold=args.confidence_threshold, invalid_value=0.0)
            print(f"  After confidence filter (threshold={args.confidence_threshold}): min={disp.min():.4f}, max={disp.max():.4f}, mean={disp.mean():.4f}")
            print(f"    Pixels remaining: {(disp > 0).sum()} ({(disp > 0).sum() / disp.size:.4f})")
        
        mask = left_right_consistency_check(disp, disp_r, max_diff=args.lr_threshold)
        print(f"  LR consistency mask (threshold={args.lr_threshold}): {mask.sum()} valid pixels out of {mask.size} ({mask.sum()/mask.size:.4f})")
        disp = np.where(mask, disp, 0.0)
        print(f"  After LR check: min={disp.min():.4f}, max={disp.max():.4f}, mean={disp.mean():.4f}")
        
        disp = speckle_filter(disp, valid_mask=mask, max_speckle_size=args.speckle_size)
        print(f"  After speckle filter (size={args.speckle_size}): min={disp.min():.4f}, max={disp.max():.4f}, mean={disp.mean():.4f}")
        
        # Improved hole filling: bilateral-weighted interpolation guided by image
        disp = fill_invalid_bilateral(
            disp, 
            guide_image=left_rect, 
            invalid_value=0.0,
            window_size=15,
            sigma_space=15.0,
            sigma_color=0.1,
            max_iterations=20,
        )
        print(f"  After bilateral hole filling: min={disp.min():.4f}, max={disp.max():.4f}, mean={disp.mean():.4f}")
        
        # Apply weighted median filter to reduce horizontal streaks (edge-preserving)
        disp = weighted_median_filter(
            disp,
            guide_image=left_rect,
            kernel_size=5,
            sigma_space=5.0,
            sigma_color=0.1,
        )
        print(f"  After weighted median filter: min={disp.min():.4f}, max={disp.max():.4f}, mean={disp.mean():.4f}")
        
        # Final bilateral filter for smooth disparity while preserving edges
        disp = bilateral_filter_disparity(
            disp,
            guide_image=left_rect,
            kernel_size=7,
            sigma_space=7.0,
            sigma_color=0.1,
            sigma_disp=5.0,
        )
        print(f"  After bilateral filter: min={disp.min():.4f}, max={disp.max():.4f}, mean={disp.mean():.4f}")
        
        t4 = time.perf_counter()
        print(f"\nPost-processing completed in {t4 - t3:.2f}s")
    else:
        t4 = t3

    # Depth and point cloud.
    depth = disparity_to_depth(disp, fx=rig.left.fx, baseline=rig.baseline)
    print(f"\nDEBUG: Depth map")
    print(f"  Shape: {depth.shape}, Dtype: {depth.dtype}")
    print(f"  Min: {depth.min():.4f} m, Max: {depth.max():.4f} m, Mean: {depth.mean():.4f} m, Std: {depth.std():.4f} m")
    print(f"  Sample pixel at ({sample_y}, {sample_x}): depth = {depth[sample_y, sample_x]:.4f} m")
    print(f"  Fraction of pixels with depth == 0: {(depth == 0).sum() / depth.size:.4f}")
    print(f"  Fraction of pixels with depth > 100m: {(depth > 100).sum() / depth.size:.4f}")
    print(f"  Fraction of pixels with depth < 0.1m: {(depth < 0.1).sum() / depth.size:.4f}")
    
    points = depth_to_point_cloud(depth, rig.left)
    t5 = time.perf_counter()
    print(f"\nDepth + point cloud in {t5 - t4:.2f}s")
    print(f"DEBUG: Point cloud")
    print(f"  Point cloud size: {points.shape[0]} points")
    if points.shape[0] > 0:
        print(f"  Point coordinates range:")
        print(f"    X: min={points[:, 0].min():.4f}, max={points[:, 0].max():.4f}, mean={points[:, 0].mean():.4f}")
        print(f"    Y: min={points[:, 1].min():.4f}, max={points[:, 1].max():.4f}, mean={points[:, 1].mean():.4f}")
        print(f"    Z: min={points[:, 2].min():.4f}, max={points[:, 2].max():.4f}, mean={points[:, 2].mean():.4f}")
    else:
        print(f"  WARNING: Point cloud is empty!")

    # Visualization.
    print(f"\n" + "=" * 60)
    print(f"DEBUG: Final output summary")
    print(f"=" * 60)
    print(f"  Final disparity: min={disp.min():.4f}, max={disp.max():.4f}, mean={disp.mean():.4f}, std={disp.std():.4f}")
    print(f"  Final depth: min={depth.min():.4f}m, max={depth.max():.4f}m, mean={depth.mean():.4f}m, std={depth.std():.4f}m")
    print(f"  Point cloud: {points.shape[0]} points")
    print(f"=" * 60)
    
    show_disparity_map(disp, title="Disparity")
    show_depth_map(depth, title="Depth (inverse scaled)")

    # Optional PLY save.
    if args.save_ply:
        save_point_cloud_ply(points, args.save_ply)
        print(f"Saved PLY to {args.save_ply}")


def main():
    parser = argparse.ArgumentParser(description="Middlebury stereo pipeline (SGM).")
    parser.add_argument("--calib", required=True, help="Path to calib.txt")
    parser.add_argument("--left", required=True, help="Path to left image (im0.png)")
    parser.add_argument("--right", required=True, help="Path to right image (im1.png)")
    parser.add_argument("--max_disparity", type=int, default=190, help="Maximum disparity (exclusive)")
    parser.add_argument("--method", choices=["sad", "census"], default="census", help="Matching cost type")
    parser.add_argument("--census_window", type=int, default=9, help="Census transform window size (odd, default 9)")
    parser.add_argument("--normalize_cost", action="store_true", help="Normalize cost volume before SGM")
    parser.add_argument("--p1", type=float, default=0.8, help="SGM penalty P1 for small disparity changes")
    parser.add_argument("--p2", type=float, default=8.0, help="SGM penalty P2 for large disparity changes")
    parser.add_argument("--adaptive_p2", action="store_true", help="Use image-gradient-adaptive P2 penalty")
    parser.add_argument("--confidence_threshold", type=float, default=0.0, help="Confidence threshold for filtering (0 to disable)")
    parser.add_argument("--lr_threshold", type=float, default=1.5, help="Left-right consistency threshold (default 1.5)")
    parser.add_argument("--speckle_size", type=int, default=50, help="Max speckle size to remove (default 50)")
    parser.add_argument("--downscale", type=float, default=1.0, help="Downscale factor, e.g., 0.5 or 0.25")
    parser.add_argument("--postprocess", action="store_true", help="Enable LR consistency, speckle filter, hole fill")
    parser.add_argument("--save_ply", type=str, default="", help="Optional output PLY path")
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
