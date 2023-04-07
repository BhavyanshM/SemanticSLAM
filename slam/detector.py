from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2
import h5py
import os
import numpy as np


class Detector:
    def __init__(self, weights_file):
        self.model = YOLO(weights_file)

    def detect(self, frame):
        result = self.model.predict(source=frame, show=False, conf=0.5)
        return result

    