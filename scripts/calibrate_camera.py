# scripts/calibrate_camera.py
# Chessboard calibration + build undistort maps (merged: calib_camera.py + build_undistort_map.py)
#
# Usage examples:
#   python scripts/calibrate_camera.py --imgs "data/calib/chessboard/*.jpg" --pattern 9 6 --square 0.025
#   python scripts/calibrate_camera.py --imgs "data/calib/chessboard/*.png" --pattern 9 6 --square 0.025 --alpha 1.0
#
# Output:
#   outputs/calib/camera_1080p.npz  (K, dist, newK, map1, map2, w, h, rms)

import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs", type=str, required=True,
                    help='Glob pattern for chessboard images, e.g. "data/calib/chessboard/*.jpg"')
    ap.add_argument("--pattern", type=int, nargs=2, required=True,
                    metavar=("NX", "NY"),
                    help="Number of inner corners (nx ny), e.g. 9 6")
    ap.add_argument("--square", type=float, required=True,
                    help="Square size in meters, e.g. 0.025")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="getOptimalNewCameraMatrix alpha (1.0 keep full FOV, 0 crop)")
    ap.add_argument("--out", type=str, default="outputs/calib/camera_1080p.npz",
                    help="Output .npz path")
    ap.add_argument("--show", action="store_true",
                    help="Visualize detected corners during calibration")
    return ap.parse_args()


def main():
    args = parse_args()
    nx, ny = args.pattern
    pattern_size = (nx, ny)

    img_paths = sorted(glob.glob(args.imgs))
    if not img_paths:
        raise FileNotFoundError(f"No images matched: {args.imgs}")

    # Prepare object points (0,0,0), (1,0,0) ... scaled by square size
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    objp *= float(args.square)

    objpoints = []
    imgpoints = []

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    img_shape = None
    used = 0
    for p in img_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]  # (w,h)

        ok, corners = cv2.findChessboardCorners(gray, pattern_size)
        if not ok:
            continue

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)
        used += 1

        if args.show:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, pattern_size, corners2, ok)
            cv2.imshow("chessboard", vis)
            cv2.waitKey(100)

    if args.show:
        cv2.destroyAllWindows()

    if used < 10:
        raise RuntimeError(f"Too few valid frames ({used}). Re-capture chessboard images.")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

    w, h = img_shape
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=float(args.alpha))
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, newK, (w, h), m1type=cv2.CV_16SC2)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path,
        K=K,
        dist=dist,
        newK=newK,
        map1=map1,
        map2=map2,
        w=w,
        h=h,
        rms=float(rms),
        pattern=np.array([nx, ny], dtype=int),
        square=float(args.square),
        alpha=float(args.alpha),
    )

    print(f"[OK] Saved: {out_path}")
    print(f"RMS reprojection error (px): {rms:.4f}")
    print("K:\n", K)
    print("dist:", dist.ravel())
    print("newK:\n", newK)


if __name__ == "__main__":
    main()
