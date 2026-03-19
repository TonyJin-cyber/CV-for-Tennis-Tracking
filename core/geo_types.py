from dataclasses import dataclass
from typing import List, Tuple

ImgPt = Tuple[float, float]     # (u, v)图像坐标bbox
BodyPt = Tuple[float, float]    # (x_b, y_b)机器人底盘平面坐标
GlobalPt = Tuple[float, float]  # (x_g, y_g)全局坐标

@dataclass(frozen=True)
class Pose2D:
    x: float      # x_r in global
    y: float      # y_r in global
    yaw: float    # theta (rad)机器人在地面平面内的朝向角

@dataclass(frozen=True)
class Detection:
    uv: ImgPt
    conf: float
