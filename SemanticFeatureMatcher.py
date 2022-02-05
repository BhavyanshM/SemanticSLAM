import numpy as np
from utils import *
np.set_printoptions(suppress=True, precision=2, linewidth=np.inf)

class SemanticFeatureMatcher:
    def __init__(self):
        self.features = []
        self.frameCount = 0
        self.table = None
        self.matches = None

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


    def plot_associations(self, img, objects):
        for i in range(len(self.features)):
            if self.table[i, self.matches[i]] != 0:
                print(i, len(self.features))
                cv2.line(img, (self.features[i].bbox[1], self.features[i].bbox[2]), (objects[self.matches[i]].bbox[1], int(img.shape[0]/2) + objects[self.matches[i]].bbox[2]), (255,0,255), 3)

        print(self.matches)
