# core/heatmap.py
import numpy as np
from typing import List, Tuple

GlobalPt = Tuple[float, float]

class Heatmap:
    def __init__(
        self,
        x_range=(-5.0, 5.0),
        y_range=(-3.0, 3.0),
        resolution=0.1,
        lambda_decay=0.95,
        beta=0.9,
    ):
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.res = resolution

        self.x_bins = int((self.x_max - self.x_min) / resolution)
        self.y_bins = int((self.y_max - self.y_min) / resolution)

        self.short = np.zeros((self.x_bins, self.y_bins), dtype=np.float32)
        self.long  = np.zeros_like(self.short)

        self.lambda_decay = lambda_decay
        self.beta = beta

    def update_short(self, points_g: List[GlobalPt]):
        self.short *= self.lambda_decay
        for x, y in points_g:
            ix = int((x - self.x_min) / self.res)
            iy = int((y - self.y_min) / self.res)
            if 0 <= ix < self.x_bins and 0 <= iy < self.y_bins:
                self.short[ix, iy] += 1.0

    def update_long(self):
        self.long = self.beta * self.long + (1.0 - self.beta) * self.short

    def get_peak(self) -> GlobalPt:
        ix, iy = np.unravel_index(np.argmax(self.long), self.long.shape)
        x = self.x_min + ix * self.res
        y = self.y_min + iy * self.res
        return (x, y)

    def suppress_region(self, center: GlobalPt, radius=0.3, factor=0.2):
        cx, cy = center
        r2 = radius * radius
        for ix in range(self.x_bins):
            for iy in range(self.y_bins):
                x = self.x_min + ix * self.res
                y = self.y_min + iy * self.res
                if (x - cx) ** 2 + (y - cy) ** 2 <= r2:
                    self.long[ix, iy] *= factor
