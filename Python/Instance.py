import numpy as np
import math


class Instance:
    def __init__(self, id, bbox, cls, conf):
        self.id = id
        self.class_conf = conf
        self.class_id = cls
        self.xyxy = bbox
        self.position = np.array([0, 0, 0])
        self.orientation = np.array([0, 0, 0])
        self.lidar = None
        self.center = None
        self.frames = []

    def extract_bbox_lidar(self, cam_points, cam_cloud):
        u, v, z = cam_points
        mx, my = (self.xyxy[2] - self.xyxy[0]) / 16, (self.xyxy[3] - self.xyxy[1]) / 16
        mx, my = 0 , 0
        u_out = np.logical_or(u < self.xyxy[0] + mx, u > self.xyxy[2] - mx)
        v_out = np.logical_or(v < self.xyxy[1] + my, v > self.xyxy[3] - my)
        outlier = np.logical_or(u_out, v_out)
        points = np.delete(cam_cloud.T, np.where(outlier), axis=1)
        self.center = np.mean(points, axis=1)
        self.lidar = points.T
        mean_mat = np.repeat(np.array([self.center]), self.lidar.shape[0], axis=0)
        distances = np.linalg.norm(mean_mat - self.lidar, axis=1)
        dist_stdev = np.std(distances)
        self.lidar = np.delete(self.lidar, np.where(distances > 0.75 * dist_stdev), axis=0)
