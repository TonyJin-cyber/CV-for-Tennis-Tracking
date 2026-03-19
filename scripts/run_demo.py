import cv2
import numpy as np
import math
from ultralytics import YOLO

# ======================
# 配置
# ======================
CAM_ID = 0
CONF_THRES = 0.35

WEIGHTS = "../models/yolo11/yolo11_tennis.pt"
CAMERA_NPZ = "../outputs/calib/camera_1080p.npz"
H_NPZ = "../outputs/calib/H_und.npz"


# ======================
# 几何工具函数
# ======================
def undistort_point(u, v, K, dist, newK):
    """
    对单个像素点去畸变
    输入/输出均为像素坐标
    """
    pts = np.array([[[u, v]]], dtype=np.float32)
    pts_und = cv2.undistortPoints(pts, K, dist, P=newK)
    return float(pts_und[0, 0, 0]), float(pts_und[0, 0, 1])


def pixel_to_ground(H, u, v):
    """
    去畸变后的像素坐标 -> 地面坐标
    """
    p = np.array([u, v, 1.0], dtype=float)
    q = H @ p
    if abs(q[2]) < 1e-9:
        return None
    return float(q[0] / q[2]), float(q[1] / q[2])


def dist2d(x, y):
    return math.sqrt(x * x + y * y)


# ======================
# 主程序
# ======================
def main():
    # ---------- 加载模型 ----------
    model = YOLO(WEIGHTS)

    # ---------- 加载相机标定 ----------
    cam = np.load(CAMERA_NPZ)
    K = cam["K"]
    dist = cam["dist"]
    newK = cam["newK"] if "newK" in cam else K

    # ---------- 加载 Homography（npz） ----------
    H = np.load(H_NPZ)["H"]

    # ---------- 打开摄像头 ----------
    cap = cv2.VideoCapture(CAM_ID)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    cv2.namedWindow("YOLO + Homography Demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO + Homography Demo", 1280, 720)

    selected_idx = 0

    print("\n[p] 切换目标   [q] 退出\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # ======================
        # 1. YOLO 在 raw 图像上检测
        # ======================
        results = model.predict(frame, conf=CONF_THRES, verbose=False)[0]

        detections = []

        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), c in zip(boxes, confs):
                u_raw = 0.5 * (x1 + x2)
                v_raw = 0.5 * (y1 + y2)

                # ======================
                # 2. 单点去畸变
                # ======================
                u_und, v_und = undistort_point(
                    u_raw, v_raw, K, dist, newK
                )

                # ======================
                # 3. Homography
                # ======================
                g = pixel_to_ground(H, u_und, v_und)
                if g is None:
                    continue

                gx, gy = g
                d = dist2d(gx, gy)

                detections.append({
                    "raw_px": (int(u_raw), int(v_raw)),
                    "g": (gx, gy),
                    "dist": d
                })

        # ======================
        # 4. 最近目标排序
        # ======================
        detections.sort(key=lambda x: x["dist"])
        if selected_idx >= len(detections):
            selected_idx = 0

        # ======================
        # 5. 可视化
        # ======================
        vis = frame.copy()

        for i, det in enumerate(detections):
            cx, cy = det["raw_px"]
            color = (0, 255, 0)
            if i == selected_idx:
                color = (0, 0, 255)
            cv2.circle(vis, (cx, cy), 6, color, -1)

        # ======================
        # 6. 输出当前目标坐标
        # ======================
        if detections:
            tgt = detections[selected_idx]
            gx, gy = tgt["g"]

            text = f"Target[{selected_idx}]  x={gx:.3f}  y={gy:.3f}"
            cv2.putText(
                vis, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 255), 2
            )

            print(
                f"\rTarget[{selected_idx}]  "
                f"ground = ({gx:.3f}, {gy:.3f})",
                end=""
            )

        cv2.imshow("YOLO + Homography Demo", vis)

        # ======================
        # 7. 键盘
        # ======================
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            if detections:
                selected_idx = (selected_idx + 1) % len(detections)

    print("\nExit.")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
