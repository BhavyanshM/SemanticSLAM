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

for file in kitti_imgs:
    img = cv2.imread(kitti_path + file)

    detections = load_detections(kitti_detections_path + file.replace('image_0', 'detections_0').replace('.png', '.txt'))

    plot_detection_boxes(img, detections, 2)

    mot.associate_detections(detections)

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", img.shape[1]*2, img.shape[0]*2)
    cv2.imshow("Image", img)
    code = cv2.waitKeyEx(100)
    if code == 113:
        exit()

