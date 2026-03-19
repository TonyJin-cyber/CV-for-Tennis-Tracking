# scripts/run_camera_pipeline.py
#
# Revised pipeline:
# raw -> YOLO detect (raw)
# -> bbox center (u_raw, v_raw)
# -> undistortPoints
# -> Homography -> (x, y)
# -> LandmarkMap + Heatmap
# -> Target selection
# -> Simulated pick via Matplotlib key event ('p')
#run :  python -m scripts.run_camera_pipeline --camera_npz outputs/calib/camera_1080p.npz --H_npz outputs/calib/H_und.npz --weights models/yolo11/yolo11_tennis.pt --cam 0 --conf 0.15 --show

import argparse
from typing import List, Tuple
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

from core.landmark_map import LandmarkMap, Observation
from core.heatmap import Heatmap


# ------------------------------
# Argument parsing
# ------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera_npz", type=str, required=True)
    ap.add_argument("--H_npz", type=str, required=True)
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--show", action="store_true")
    return ap.parse_args()


# ------------------------------
# Homography mapping
# ------------------------------
def uv_to_xy(H: np.ndarray, u: float, v: float) -> Tuple[float, float]:
    p = np.array([u, v, 1.0], dtype=float)
    q = H @ p
    if abs(q[2]) < 1e-9:
        return (float("nan"), float("nan"))
    return (q[0] / q[2], q[1] / q[2])


# ------------------------------
# YOLO detector wrapper
# ------------------------------
class Detector:
    def __init__(self, weights: str):
        from ultralytics import YOLO
        self.model = YOLO(weights)

    def detect(self, img_bgr: np.ndarray, conf: float):
        results = self.model.predict(source=img_bgr, conf=conf, verbose=False)
        out = []
        r = results[0]
        if r.boxes is None:
            return out
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c in zip(boxes, confs):
            out.append((float(x1), float(y1), float(x2), float(y2), float(c)))
        return out


# ------------------------------
# Main pipeline
# ------------------------------
def main():
    args = parse_args()

    # ---- Load camera calibration ----
    cam_npz = np.load(args.camera_npz)
    K = cam_npz["K"]
    dist = cam_npz["dist"]
    newK = cam_npz.get("newK", K)

    # ---- Load homography ----
    H = np.load(args.H_npz)["H"]

    # ---- Initialize modules ----
    detector = Detector(args.weights)

    landmark_map = LandmarkMap(
        assoc_dist=0.25,
        pos_ema_alpha=0.7,
        init_min_conf=0.3,
    )

    heatmap = Heatmap(
        x_range=(-5.0, 5.0),
        y_range=(-3.0, 3.0),
        resolution=0.1,
        lambda_decay=0.95,
        beta=0.9,
    )

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.cam}")

    cv2.namedWindow("pipeline (raw)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("pipeline (raw)", 1280, 720)

    # ---- Matplotlib pick event flag ----
    pick_request = False

    def on_key(event):
        nonlocal pick_request
        if event.key == "p":
            pick_request = True

    plt.figure(1)
    plt.connect("key_press_event", on_key)

    last_long_update = time.time()
    last_target = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # =================================
        # 1. YOLO detection on RAW image
        # =================================
        boxes = detector.detect(frame, conf=args.conf)

        obs_list: List[Observation] = []
        vis_boxes = []

        for x1, y1, x2, y2, c in boxes:
            u_raw = 0.5 * (x1 + x2)
            v_raw = 0.5 * (y1 + y2)

            # ---- undistort POINT only ----
            pts = np.array([[[u_raw, v_raw]]], dtype=np.float32)
            pts_und = cv2.undistortPoints(pts, K, dist, P=newK)
            u_und, v_und = pts_und[0, 0]

            # ---- Homography ----
            x, y = uv_to_xy(H, u_und, v_und)

            if not np.isnan(x) and not np.isnan(y):
                obs_list.append(Observation(pos_g=(x, y), conf=c))

            vis_boxes.append((x1, y1, x2, y2, c))

        # =================================
        # 2. Update maps
        # =================================
        landmark_map.update(obs_list)
        heatmap.update_short([obs.pos_g for obs in obs_list])

        now = time.time()
        if now - last_long_update > 0.5:
            heatmap.update_long()
            last_long_update = now

        # =================================
        # 3. Target selection
        # =================================
        robot_pos_g = (0.0, 0.0)
        target, source = landmark_map.choose_next_target_with_heatmap(
            robot_pos_g, heatmap, min_landmarks=3
        )

        if target != last_target:
            print(f"[INFO] New target: {target} (source={source})")
            last_target = target

        target_xy = None
        if target is not None:
            if source == "landmark":
                lm = landmark_map.get(target)
                if lm is not None:
                    target_xy = lm.pos_g
            else:
                target_xy = target

        # =================================
        # 4. Simulated pick (Matplotlib key)
        # =================================
        if pick_request and target_xy is not None:
            print("[PICK] Simulated pick event")

            if source == "landmark":
                landmark_map.mark_picked(target)
                picked_center = target_xy
            else:
                picked_center = target_xy

            heatmap.suppress_region(
                center=picked_center,
                radius=0.4,
                factor=0.2,
            )

            pick_request = False

        # ===== 立即重新选择目标 =====
        target, source = landmark_map.choose_next_target_with_heatmap(
            robot_pos_g, heatmap, min_landmarks=3
        )

        # 重新解析 target_xy
        if target is not None:
            if source == "landmark":
                lm = landmark_map.get(target)
                target_xy = lm.pos_g if lm is not None else None
            else:
                target_xy = target

        # =================================
        # 5. Visualization
        # =================================
        if args.show:
            vis = frame.copy()
            for x1, y1, x2, y2, c in vis_boxes:
                cv2.rectangle(
                    vis,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    vis,
                    f"{c:.2f}",
                    (int(x1), max(0, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
            cv2.imshow("pipeline (raw)", vis)

            # ---- Top-down map ----
            plt.clf()
            # plt.imshow(
            #     heatmap.long.T,
            #     origin="lower",
            #     extent=[-5, 5, -3, 3],
            #     cmap="hot",
            #     alpha=0.6,
            # )

            lms = landmark_map.active()
            if lms:
                xs = [lm.pos_g[0] for lm in lms]
                ys = [lm.pos_g[1] for lm in lms]
                plt.scatter(xs, ys, c="cyan", s=20, label="Landmarks")

            if target_xy is not None:
                plt.scatter(
                    [target_xy[0]],
                    [target_xy[1]],
                    c="blue",
                    marker="*",
                    s=120,
                    label="Target",
                )

            plt.legend()
            plt.pause(0.001)

        # ---- exit ----
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
