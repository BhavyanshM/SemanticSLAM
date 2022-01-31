import cv2
import os
import numpy as np

from KITTI_Classes import *
from utils import *
from MultiObjectTracker import *

kitti_path = '/home/quantum/Workspace/Storage/Other/Temp/dataset/sequences/00/image_0/'
kitti_detections_path = '/home/quantum/Workspace/Storage/Other/Temp/dataset/sequences/00/detections_0/'

kitti_imgs = sorted(os.listdir(kitti_path))

mot = MultiObjectTracker()

detections = load_detections(kitti_detections_path + kitti_imgs[0].replace('image_0', 'detections_0').replace('.png', '.txt'))
objects = create_instances(detections)
mot.initialize_tracks(objects)

prevImg = cv2.imread(kitti_path + kitti_imgs[0])


for file in kitti_imgs:
    img = cv2.imread(kitti_path + file)

    detections = load_detections(kitti_detections_path + file.replace('image_0', 'detections_0').replace('.png', '.txt'))
    objects = create_instances(detections)

    mot.associate_detections(objects)


    plot_detection_boxes(img, objects, 2)
    combined = combine_images_vertical(prevImg, img)
    mot.plot_associations(combined, objects)
    display(combined)
    prevImg = img

    mot.tracks = objects
    # images = [prevImg, img]
    # final = cv2.vconcat(images)
    # cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Image", final.shape[1]*2, final.shape[0]*2)
    # cv2.imshow("Image", final)
    # code = cv2.waitKeyEx(50)
    # if code == 113:
    #     cv2.waitKey(0)
    #     exit()

