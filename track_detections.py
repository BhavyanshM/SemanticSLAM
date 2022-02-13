import cv2
import os
import numpy as np

from KITTI_Classes import *
from utils import *
from SemanticFeatureMatcher import *
from TransformUtils import *

from Open3DRenderer import *

class TrackingApp:
    def __init__(self):

        self.kitti_path = '/home/quantum/Workspace/Storage/Other/Temp/dataset/sequences/00/image_0/'
        self.kitti_detections_path = '/home/quantum/Workspace/Storage/Other/Temp/dataset/sequences/00/detections_0/'
        self.kitti_poses_path = '/home/quantum/Workspace/Storage/Other/Temp/dataset/data_odometry_poses/poses/00.txt'

        self.data_exists = os.path.isdir(self.kitti_path)

        self.kitti_imgs = sorted(os.listdir(self.kitti_path)) if self.data_exists else None
        self.detections = load_detections(self.kitti_detections_path + self.kitti_imgs[0].replace('image_0', 'detections_0').replace('.png', '.txt')) if self.data_exists else []
        self.prevImg = cv2.imread(self.kitti_path + self.kitti_imgs[0]) if self.data_exists else None
        self.gt = np.loadtxt(self.kitti_poses_path, delimiter=' ') if self.data_exists else None

        print(self.gt.shape)

        self.objects = create_tracks(self.detections)
        self.matcher = SemanticFeatureMatcher()
        self.matcher.initialize_tracks(self.objects)
        self.renderer = Open3DRenderer()

        self.pose = np.eye(4)
        self.delta_pose = get_rotation_y(0.4)
        self.delta_pose[0, 3] = 5
        self.delta_pose[2, 3] = 0



    def run(self):
        for file in self.kitti_imgs:
            img = cv2.imread(self.kitti_path + file)

            detections = load_detections(self.kitti_detections_path + file.replace('image_0', 'detections_0').replace('.png', '.txt'))
            objects = create_tracks(detections)

            self.matcher.associate_detections(objects)

            plot_detection_boxes(img, objects, 2, self.matcher.classes)
            combined = combine_images_vertical(self.prevImg, img)

            plot_associations(combined, self.matcher, objects)
            display(combined)
            self.prevImg = img

            self.matcher.features = objects

            # self.pose = self.pose @ self.delta_pose
            #
            # self.renderer.submit_pose(self.pose)
            # self.renderer.update()


    def run_no_data(self):
        gt = np.loadtxt('/home/quantum/Workspace/Storage/Other/Temp/dataset/data_odometry_poses/poses/00.txt', delimiter=' ')
        i = 0
        while True:
            pose = np.eye(4)
            pose[:3,:4] = gt[i].reshape((3,4))
            print(pose)

            self.renderer.submit_pose(pose)

            self.renderer.update()
            i+=1

    def init_experimental(self):
        p1 = (np.array([3, 0, 4]), np.array([-4.1, 0, 3]))
        p2 = (np.array([-3, 0, 4]), np.array([4.1, 0, 3]))
        p3 = (np.array([0, 1, 0]), np.array([0, 1, 1]))

        ps1, ps2, points = triangulate_convex_polytope((0, 300, 250, 180, 100), (0, 200, 150, 120, 80), self.delta_pose)

        print(ps1)


        self.renderer.submit_quad(np.array([0,0,0]), ps1[0][:3], 5.0, 1.0, [0.3, 0.4, 0.6])
        self.renderer.submit_quad(np.array([0,0,0]), ps1[1][:3], 5.0, 1.0, [0.6, 0.7, 0.3])

        self.renderer.submit_quad(ps2[0][:3], ps2[0][3:], 5.0, 1.0, [0.5, 0.8, 0.3])
        self.renderer.submit_quad(ps2[1][:3], ps2[1][3:], 5.0, 1.0, [0.3, 0.7, 0.5])

        print("Shape Points:", points.shape)


        for i in range(points.shape[0]):
            self.renderer.submit_sphere(points[i])


        self.renderer.submit_polytope(points)

        # self.renderer.submit_quad(p1[0], p1[1], 4.0, 1.0, [0.3, 0.4, 0.6])
        # self.renderer.submit_quad(p2[0], p2[1], 4.0, 1.0, [0.3, 0.4, 0.6])
        # self.renderer.submit_quad(p3[0], p3[1], 2.5, 2.5, [0.7, 0.6, 0.6])

        # x = find_plane_intersection(get_plane(p1[0], p1[1]),
        #                             get_plane(p2[0], p2[1]),
        #                             get_plane(p3[0], p3[1]))

        # self.renderer.submit_quad(get_plane(np.array([0, 0, 0]), np.array([-1, 0, 1])), 2.0, 0.3, 0.4, 0.6)
        #
        # self.renderer.submit_quad(get_plane(np.array([10, 1, -2]), np.array([0, 0, 1])), 2.0, 0.5, 0.7, 0.3)
        # self.renderer.submit_quad(get_plane(np.array([10, -1, -1]), np.array([0, 0, 1])), 2.0, 0.7, 0.3, 0.6)

        # self.matcher.triangulate_convex_polytope(None, None)

if __name__ == "__main__":
    app = TrackingApp()
    # app.init_experimental()
    app.run_no_data()