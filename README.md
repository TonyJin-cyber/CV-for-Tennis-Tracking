# CV-for-Tennis-Tracking

一个用于**网球检测 + 地面坐标映射 + 目标选择**的计算机视觉实验代码库。项目核心流程是：

> 原始相机画面 → YOLO 检测 → 单点去畸变 → 单应映射到地面坐标 → 地标聚合与去重 → 热力图补偿选点

---

## 1. 项目目标

本项目面向“在球场地面上定位网球并为后续机器人拾取提供目标点”的场景，覆盖：

- 相机标定（内参与畸变模型）
- 地面单应标定（像素到米制平面坐标）
- 实时检测与坐标变换
- 多帧地标融合（LandmarkMap）
- 热力图辅助选点（Heatmap）

---

## 2. 目录结构

```text
CV-for-Tennis-Tracking/
├── core/
│   ├── geo_types.py          # 类型定义（Detection / Pose2D 等）
│   ├── heatmap.py            # 短期/长期热力图
│   └── landmark_map.py       # 地标数据关联、状态管理、目标选择
├── modules/
│   ├── detect_yolo.py        # YOLO 检测接口（当前仍有 TODO）
│   ├── geo_homography.py     # 图像点 -> 机体平面点
│   └── geo_se2.py            # 机体坐标 -> 全局坐标
├── scripts/
│   ├── calibrate_camera.py   # 棋盘格标定，输出去畸变参数
│   ├── calibrate_homography.py # 点选地面点，估计单应矩阵
│   ├── run_camera_pipeline.py  # 主实时流水线
│   ├── run_demo.py             # 独立 demo（较早期）
│   └── test.py                 # 与主流水线高度相似的实验脚本
├── data/calib/               # 标定图片和地面点数据
├── outputs/calib/            # 标定输出（camera_1080p.npz / H_und.npz）
└── models/yolo11/            # 模型权重
```

---

## 3. 环境依赖

建议 Python 3.10+。核心依赖：

- `opencv-python`
- `numpy`
- `matplotlib`
- `ultralytics`

示例安装：

```bash
pip install opencv-python numpy matplotlib ultralytics
```

> 说明：仓库当前没有 `requirements.txt`，建议后续补充以固定版本。

---

## 4. 快速开始

### 4.1 相机标定

```bash
python scripts/calibrate_camera.py \
  --imgs "data/calib/chessboard/*.png" \
  --pattern 9 6 \
  --square 0.025 \
  --out outputs/calib/camera_1080p.npz
```

### 4.2 单应标定（地面映射）

```bash
python scripts/calibrate_homography.py \
  --camera_npz outputs/calib/camera_1080p.npz \
  --img data/calib/ground_points/ground_raw_1080p.png \
  --xy_csv data/calib/ground_points/xy_16.csv \
  --out outputs/calib/H_und.npz
```

### 4.3 运行主流水线

```bash
python -m scripts.run_camera_pipeline \
  --camera_npz outputs/calib/camera_1080p.npz \
  --H_npz outputs/calib/H_und.npz \
  --weights models/yolo11/yolo11_tennis.pt \
  --cam 0 \
  --conf 0.25 \
  --show
```

运行中：

- `q` / `ESC`：退出
- 在 Matplotlib 窗口按 `p`：触发一次模拟拾取并抑制目标周围热力图

---

## 5. 核心模块说明

### 5.1 LandmarkMap（`core/landmark_map.py`）

- 使用最近邻 + 距离门限进行观测关联
- 对地标位置做 EMA 更新
- 状态分为 `active` 与 `picked`
- 支持按区域标记已拾取、按最近距离选择目标
- 当活跃地标较少时，可切换到热力图峰值策略

### 5.2 Heatmap（`core/heatmap.py`）

- `short`：短期累积（随时间衰减）
- `long`：长期平滑
- `get_peak()` 给出当前峰值候选点
- `suppress_region()` 在模拟拾取后对局部降权

### 5.3 几何映射

- `scripts/run_camera_pipeline.py` 中通过 `cv2.undistortPoints` 做单点去畸变
- 再通过单应矩阵 `H` 将像素点映射到地面平面 `(x,y)`（米）

---


