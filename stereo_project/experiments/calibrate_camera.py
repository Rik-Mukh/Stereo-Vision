"""
Camera calibration from checkerboard images using OpenCV.
"""

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np

from stereo_project.core.camera import CameraIntrinsics


def calibrate_camera(
    image_paths: list[str],
    checkerboard_size: tuple[int, int],
    square_size_mm: float,
    show_corners: bool = False,
) -> tuple[CameraIntrinsics, np.ndarray, float]:
    """
    Calibrate a single camera using checkerboard images.

    checkerboard_size: (cols, rows) of inner corners (e.g., 7x5 for an 8x6 board).
    square_size_mm: length of a checker square in millimeters.
    """
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0 : checkerboard_size[0], 0 : checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm / 1000.0  # convert to meters

    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    successes = 0

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"[warn] Could not read {path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        found, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        if not found:
            print(f"[warn] Corners not found in {Path(path).name}")
            continue

        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )
        objpoints.append(objp)
        imgpoints.append(corners_refined)
        successes += 1

        if show_corners:
            vis = cv2.drawChessboardCorners(img.copy(), checkerboard_size, corners_refined, found)
            cv2.imshow("Corners", vis)
            cv2.waitKey(200)

    if show_corners:
        cv2.destroyAllWindows()

    print(f"[info] Detected corners in {successes}/{len(image_paths)} images")
    if successes < 10:
        raise RuntimeError("Need at least 10 successful detections for stable calibration.")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        (w, h),
        None,
        None,
        flags=cv2.CALIB_FIX_K3,
    )
    if not ret:
        raise RuntimeError("Calibration failed.")

    # Distortion coefficients can be length 1, 4, 5, or 8 depending on flags.
    dist_flat = dist_coeffs.ravel()
    dist_padded = np.zeros(5, dtype=float)
    dist_padded[: min(len(dist_flat), 5)] = dist_flat[: min(len(dist_flat), 5)]

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    intrinsics = CameraIntrinsics(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=w,
        height=h,
        k1=dist_padded[0],
        k2=dist_padded[1],
        p1=dist_padded[2],
        p2=dist_padded[3],
        k3=dist_padded[4],
    )

    # Reprojection error (RMS)
    mean_error = 0.0
    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        error = cv2.norm(imgpoints[i], projected, cv2.NORM_L2) / len(projected)
        mean_error += error
    reproj_error = mean_error / len(objpoints)

    return intrinsics, dist_coeffs, reproj_error


def save_intrinsics(intrinsics: CameraIntrinsics, output_path: str) -> None:
    """Save intrinsics to a simple text file: fx fy cx cy."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{intrinsics.fx} {intrinsics.fy} {intrinsics.cx} {intrinsics.cy}\n")
    print(f"[info] Saved intrinsics to {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate camera from checkerboard images.")
    parser.add_argument(
        "images",
        nargs="+",
        help="Checkerboard images (paths or glob patterns).",
    )
    parser.add_argument("--cols", type=int, default=7, help="Inner corners across (default 7 for 8x6 board).")
    parser.add_argument("--rows", type=int, default=5, help="Inner corners down (default 5 for 8x6 board).")
    parser.add_argument("--square_size", type=float, default=30.0, help="Square size in millimeters.")
    parser.add_argument("--show_corners", action="store_true", help="Visualize detected corners.")
    parser.add_argument("--output", type=str, default="camera_intrinsics.txt", help="Output intrinsics file.")
    return parser.parse_args()


def main():
    args = _parse_args()

    image_paths: list[str] = []
    for pattern in args.images:
        expanded = glob.glob(pattern)
        if expanded:
            image_paths.extend(expanded)
        else:
            image_paths.append(pattern)

    if not image_paths:
        raise ValueError("No images found. Check your paths/globs.")

    print(f"[info] Found {len(image_paths)} images")
    print(f"[info] Checkerboard inner corners: {args.cols} x {args.rows}")
    print(f"[info] Square size: {args.square_size} mm")

    intrinsics, dist_coeffs, reproj_error = calibrate_camera(
        image_paths,
        (args.cols, args.rows),
        args.square_size,
        show_corners=args.show_corners,
    )

    print("\n===== Calibration Results =====")
    print(f"Image size: {intrinsics.width} x {intrinsics.height}")
    print(f"fx: {intrinsics.fx:.3f}, fy: {intrinsics.fy:.3f}")
    print(f"cx: {intrinsics.cx:.3f}, cy: {intrinsics.cy:.3f}")
    print("Distortion coefficients:", dist_coeffs.ravel().tolist())
    print(f"Reprojection error (RMS): {reproj_error:.4f} px")

    save_intrinsics(intrinsics, args.output)
    print("\nUse with run_sfm_stereo_pair.py via: --intrinsics", args.output)


if __name__ == "__main__":
    main()

