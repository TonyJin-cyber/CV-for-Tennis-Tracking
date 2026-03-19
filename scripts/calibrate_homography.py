# scripts/calibrate_homography.py
# Compute homography H using points clicked on UNDISTORTED image.
#
# Usage example:
#   python scripts/calibrate_homography.py \
#       --camera_npz outputs/calib/camera_1080p.npz \
#       --img data/calib/ground_points/ground_raw_1080p.jpg \
#       --xy_csv data/calib/ground_points/xy_16.csv \
#       --out outputs/calib/H_und.npz


import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera_npz", type=str, required=True, help="outputs/calib/camera_1080p.npz")
    ap.add_argument("--img", type=str, required=True, help="Ground points raw image (1080p)")
    ap.add_argument("--xy_csv", type=str, required=True, help="CSV of (x,y) in meters, one per line")
    ap.add_argument("--out", type=str, default="outputs/calib/H_und.npz", help="Output .npz path")
    ap.add_argument("--ransac_thresh", type=float, default=3.0, help="RANSAC reproj threshold (px)")
    return ap.parse_args()


def load_xy(csv_path: str) -> np.ndarray:
    # Accept with or without header; ignore empty lines.
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if any(k in s.lower() for k in ["x", "y"]) and "," in s:
                # likely header
                continue
            parts = [p.strip() for p in s.split(",")]
            if len(parts) < 2:
                continue
            rows.append([float(parts[0]), float(parts[1])])
    XY = np.array(rows, dtype=np.float32)
    if XY.shape[0] < 4:
        raise RuntimeError(f"XY points <4 from {csv_path}")
    return XY


def main():
    args = parse_args()

    cam = np.load(args.camera_npz)
    map1, map2 = cam["map1"], cam["map2"]

    raw = cv2.imread(args.img)
    if raw is None:
        raise FileNotFoundError(f"Cannot read image: {args.img}")

    und = cv2.remap(raw, map1, map2, interpolation=cv2.INTER_LINEAR)

    XY = load_xy(args.xy_csv)

    clicked = []
    und_show = und.copy()

    def on_mouse(event, x, y, flags, param):
        nonlocal und_show
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append([x, y])
            cv2.circle(und_show, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(
                und_show, str(len(clicked)),
                (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2, cv2.LINE_AA
            )
            print(f"clicked {len(clicked)}: u={x}, v={y}")

    print("Instructions:")
    print("  - Left click to select points on UNDISTORTED image.")
    print("  - Press 'c' to clear and restart clicking.")
    print("  - Press 'q' or ESC when done.")
    print(f"Need clicks = {XY.shape[0]} (must match xy_csv rows)")

    cv2.namedWindow("undistorted", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("undistorted", on_mouse)

    while True:
        cv2.imshow("undistorted", und_show)
        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord("q")):  # ESC or q
            break
        if k == ord("c"):
            clicked.clear()
            und_show = und.copy()
            print("cleared.")

    cv2.destroyAllWindows()

    UV = np.array(clicked, dtype=np.float32)
    if UV.shape[0] != XY.shape[0]:
        raise RuntimeError(f"Clicked UV count {UV.shape[0]} != XY count {XY.shape[0]}")

    H, inliers = cv2.findHomography(UV, XY, method=cv2.RANSAC, ransacReprojThreshold=float(args.ransac_thresh))
    if H is None:
        raise RuntimeError("findHomography failed (H is None)")

    inl = int(inliers.sum()) if inliers is not None else 0
    print(f"Inliers: {inl}/{len(UV)}")

    # Quick numeric check (reprojection error in meters)
    UV_h = np.concatenate([UV, np.ones((UV.shape[0], 1), dtype=np.float32)], axis=1)  # Nx3
    XY_pred_h = (H @ UV_h.T).T
    XY_pred = XY_pred_h[:, :2] / XY_pred_h[:, 2:3]
    err = np.linalg.norm(XY_pred - XY, axis=1)
    print(f"Reproj error (meters): mean={err.mean():.4f}, max={err.max():.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, H=H)
    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()
