import cv2
import os
import numpy as np

from KITTI_Classes import *
from utils import *
from SemanticFeatureMatcher import *

from Open3DRenderer import *

class TrackingApp:
    def __init__(self):

        self.kitti_path = '/home/quantum/Workspace/Storage/Other/Temp/dataset/sequences/00/image_0/'
        self.kitti_detections_path = '/home/quantum/Workspace/Storage/Other/Temp/dataset/sequences/00/detections_0/'
        self.kitti_imgs = sorted(os.listdir(self.kitti_path))
        self.mot = SemanticFeatureMatcher()
        self.detections = load_detections(self.kitti_detections_path + self.kitti_imgs[0].replace('image_0', 'detections_0').replace('.png', '.txt'))
        self.objects = create_tracks(self.detections)
        self.mot.initialize_tracks(self.objects)
        self.prevImg = cv2.imread(self.kitti_path + self.kitti_imgs[0])
        self.renderer = Open3DRenderer()
        self.pose = np.eye(4)
        self.delta_pose = np.eye(4)
        self.delta_pose[0, 3] = 0.1

        self.renderer.submit_quad(self.pose)

        self.mot.triangulate_convex_polytope(None, None)

    def run(self):
        for file in self.kitti_imgs:
            img = cv2.imread(self.kitti_path + file)

            detections = load_detections(self.kitti_detections_path + file.replace('image_0', 'detections_0').replace('.png', '.txt'))
            objects = create_tracks(detections)

            self.mot.associate_detections(objects)


            plot_detection_boxes(img, objects, 2, self.mot.classes)
            combined = combine_images_vertical(self.prevImg, img)

            plot_associations(combined, self.mot, objects)
            # display(combined)
            self.prevImg = img

            self.mot.features = objects

            self.pose = self.pose @ self.delta_pose

            self.renderer.submit_pose(self.pose)
            self.renderer.update()

            # images = [prevImg, img]
            # final = cv2.vconcat(images)
            # cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("Image", final.shape[1]*2, final.shape[0]*2)
            # cv2.imshow("Image", final)
            # code = cv2.waitKeyEx(50)
            # if code == 113:
            #     cv2.waitKey(0)
            #     exit()

if __name__ == "__main__":
    app = TrackingApp()
    app.run()