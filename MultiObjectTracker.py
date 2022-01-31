import numpy as np
from utils import *
np.set_printoptions(suppress=True, precision=2, linewidth=np.inf)

class MultiObjectTracker:
    def __init__(self):
        self.tracks = []
        self.frameCount = 0
        self.table = None

    def initialize_tracks(self, detections):
        self.tracks.extend(detections)

    def associate_detections(self, objects):
        print("Frames:", self.frameCount, "Detections:", len(objects), "Tracks:", len(self.tracks))
        self.table = np.zeros(shape=(len(self.tracks), len(objects)))
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(objects):
                iou = compute_iou(xywh_to_xyxy(det.bbox[1:]), xywh_to_xyxy(track.bbox[1:]))
                self.table[i,j] = iou

        print(self.table)

    def plot_associations(self, img, objects):
        indices = np.argmax(self.table, axis=1)

        for i in range(len(self.tracks)):
            if self.table[i,indices[i]] != 0:
                cv2.line(img, (self.tracks[i].bbox[1], self.tracks[i].bbox[2]), (objects[indices[i]].bbox[1], int(img.shape[0]/2) + objects[indices[i]].bbox[2]), (255,0,255), 3)

        print(indices)
