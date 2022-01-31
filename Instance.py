import numpy as np
import random


class Instance:
    def __init__(self, cls, id, initPos, timestamp, hog, bbox):
        # Initialize
        self.cls = cls
        self.last_time = timestamp
        self.prev_pos = [initPos[0], initPos[1]]

        # Update these
        self.id = id
        self.imgTrack = []
        self.timestamps = [timestamp]
        self.hog = hog
        self.bbox = bbox
        self.history = [bbox]
        self.active = True
        self.candidate = False
        self.found = True
        self.num_readings = 1
        self.num_failures = 0
        self.color = [random.randint(0,255) for _ in range(3)]

    def update(self, timestamp, measuredSeaPos, dist):
        dt = timestamp - self.last_time
        self.last_time = timestamp
        self.timestamps.append(timestamp)

    def predict(self, current_time):
        dt = current_time - self.last_time
        self.last_time = current_time

        return np.array([self.filter.x[0], self.filter.x[2]])

    def update_hog(self, hog):
        self.hog = hog

    def update_bbox(self, dims, imgPoint, bbox):
        # self.dims = [0.5 * self.dims[i] + 0.5 * dims[i] for i in range(len(dims))]
        self.dims = [0.98 * self.dims[i] + 0.02 * dims[i] for i in range(len(dims))]
        x, y, w, h = imgPoint[0], imgPoint[1], self.dims[0], self.dims[1]
        self.bbox = [x - w / 2, y - 0.9 * h, x + w / 2, y + 0.1 * h]
        self.bbox = [int(n) for n in self.bbox]
        self.history.append(self.bbox)
        # if (self.id == 0): print("Bboxes:", bbox, self.bbox, dims, self.dims)
