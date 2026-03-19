import math
from core.geo_types import BodyPt, GlobalPt, Pose2D

def body_to_global(p_b: BodyPt, pose: Pose2D) -> GlobalPt:
    x_b, y_b = p_b
    c = math.cos(pose.yaw)# pose.yaw 车体坐标系相对于全局坐标系的偏航角
    s = math.sin(pose.yaw)# 逆时针为正
    #核心：旋转矩阵
    x_g = c * x_b - s * y_b + pose.x
    y_g = s * x_b + c * y_b + pose.y
    return (x_g, y_g)
