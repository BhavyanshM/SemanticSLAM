import numpy as np
from utils import *

class MultiObjectTracker:
    def __init__(self):
        self.tracks = []

    def associate_detections(self, detections):
        if len(self.tracks) == 0:
            self.tracks.extend(detections)

        else:
            for track in self.tracks:
                for det in detections:
                    iou = compute_iou(xywh_to_xyxy(det[1:]), xywh_to_xyxy(track[1:]))
