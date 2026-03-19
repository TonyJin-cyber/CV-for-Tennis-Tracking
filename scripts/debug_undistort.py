# scripts/debug_undistort.py
# Quick visual check: raw vs undistorted (camera file must include map1/map2).
#
# Usage:
#   python scripts/debug_undistort.py --camera_npz outputs/calib/camera_1080p.npz --img test.jpg
#   python scripts/debug_undistort.py --camera_npz outputs/calib/camera_1080p.npz --cam 0

import argparse

import cv2
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera_npz", type=str, required=True)
    ap.add_argument("--img", type=str, default=None, help="single image to test")
    ap.add_argument("--cam", type=int, default=None, help="camera index to test, e.g. 0")
    return ap.parse_args()


def main():
    args = parse_args()
    cam = np.load(args.camera_npz)
    map1, map2 = cam["map1"], cam["map2"]

    if args.img:
        raw = cv2.imread(args.img)
        if raw is None:
            raise FileNotFoundError(args.img)
        und = cv2.remap(raw, map1, map2, cv2.INTER_LINEAR)

        cv2.imshow("raw", raw)
        cv2.imshow("undistorted", und)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    if args.cam is None:
        raise ValueError("Provide either --img or --cam")

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.cam}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        und = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        cv2.imshow("raw", frame)
        cv2.imshow("undistorted", und)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
