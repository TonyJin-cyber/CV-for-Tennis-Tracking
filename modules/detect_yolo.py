from typing import List
from core.geo_types import Detection
from ultralytics import YOLO
import os

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "models", "yolo11", "yolo11_tennis.pt"
)

model = YOLO(MODEL_PATH)


def bbox_center_xyxy(x1: float, y1: float, x2: float, y2: float):
    u = (x1 + x2) / 2.0
    v = (y1 + y2) / 2.0
    return (u, v)

def run_yolo(image_bgr) -> List[Detection]:
    """
    Return detections as (u, v, conf).
    Replace the TODO with your YOLO inference.
    """
    dets: List[Detection] = []

    # TODO: results = model.predict(image_bgr, ...)
    # for each box:
    #   x1,y1,x2,y2,conf = ...
    #   uv = bbox_center_xyxy(x1,y1,x2,y2)
    #   dets.append(Detection(uv=uv, conf=conf))

    return dets
