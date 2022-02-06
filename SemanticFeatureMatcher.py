import numpy as np
from utils import *
np.set_printoptions(suppress=True, precision=2, linewidth=np.inf)

class SemanticFeatureMatcher:
    def __init__(self):
        self.features = []
        self.frameCount = 0
        self.table = None
        self.matches = None
        self.classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
           'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
           'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
           'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
           'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
           'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush', 'window']

    def initialize_tracks(self, detections):
        self.features.extend(detections)

    def associate_detections(self, objects):
        print("Frames:", self.frameCount, "Detections:", len(objects), "Tracks:", len(self.features))

        self.table = np.zeros(shape=(len(self.features), len(objects)))

        for i, track in enumerate(self.features):
            for j, det in enumerate(objects):
                iou = compute_iou(xywh_to_xyxy(det.bbox[1:]), xywh_to_xyxy(track.bbox[1:]))
                self.table[i, j] = iou
                if det.cls != track.cls:
                    self.table[i,j] = 0

        print(self.table)

        self.matches = np.argmax(self.table, axis=1)

        # for i in range(len(self.features)):
        #     if self.table[i, self.matches[i]] != 0:
        #         self.features[i].append(objects[self.matches[i]])



