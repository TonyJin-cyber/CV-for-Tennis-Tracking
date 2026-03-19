import numpy as np
from core.geo_types import ImgPt, BodyPt

class HomographyMapper:#单应性矩阵映射
    def __init__(self, H: np.ndarray):#init构造函数；H是numpy数组
        assert H.shape == (3, 3)#检查矩阵形状3*3
        self.H = H.astype(float)#强制变换类型为float

    def img_to_body(self, uv: ImgPt) -> BodyPt:#把图像里的一个点，通过单应矩阵，映射到机器人地面坐标
        u, v = uv
        p = np.array([u, v, 1.0], dtype=float)#齐次坐标：让平面点能参与矩阵映射
        q = self.H @ p  #矩阵乘法
        if abs(q[2]) < 1e-9:#判断第三个元素是否接近0
            return (float("nan"), float("nan"))
        #归一化（去齐次化）：齐次 ->笛卡尔
        x_b = q[0] / q[2]
        y_b = q[1] / q[2]

        return (float(x_b), float(y_b))
