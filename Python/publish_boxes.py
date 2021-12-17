import argparse
import os
import time
from pathlib import Path

import cv2
import rospy

from sensor_msgs.msg import PointCloud2

# import open3d as o3d
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from Instance import *
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized


from registration import *

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

class YOLOv5:

    def __init__(self):
        source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size

        set_logging()
        self.device = select_device(opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.imgsz = check_img_size(imgsz, s=self.model.stride.max())  # check img_siz

        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        self.objects = []
        self.dims = (376, 1241, 3)

        self.input_image = None

        if self.half:
            self.model.half()  # to FP16

        self.bbox_publisher = rospy.Publisher("/semantic/bbox/polygons", PointCloud2, queue_size=2)
        self.image_subscriber = rospy.Subscriber("/kitti/left/image_rect/compressed", CompressedImage, self.input_callback, queue_size=1)
        self.processor = rospy.Timer(rospy.Duration(0.01), self.timer_callback)
        self.image_available = False


        # torch_image = torch.zeros((1, 3, imgsz, imgsz), device=self.device)  # init img
        # _ = self.model(torch_image.half() if self.half else torch_image) if self.device.type != 'cpu' else None  # run once

    def input_callback(self, msg):
        np_arr = np.fromstring(msg.data, np.uint8)
        self.input_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.image_available = True

    def timer_callback(self, dt):
        if self.image_available:
            image = self.input_image.copy()
            self.image_available = False

            self.detect(np.asarray(image))

    def detect(self, input_image, save_img=False):

        source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://'))


        im0s = input_image
        img = letterbox(input_image)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        time_count = 0
        t0 = time.time()
        t1 = time_synchronized()
        time_count += 1

        # TODO: Separate inferencing code into a separate function and publish results
        # TODO: Follow Reference: https://gist.github.com/lucasw/ea04dcd65bc944daea07612314d114bb

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img, augment=opt.augment)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)


        # Process detections
        total_objects = 0
        for i, det in enumerate(pred):  # detections per image

            s, im0 = '', im0s
            s += '%gx%g ' % img.shape[2:]  # print string
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, self.names[int(c)])  # add to string
                for *xyxy, conf, cls in reversed(det):
                    bbox = np.array(torch.tensor(xyxy).view(-1).tolist())
                    total_objects += 1

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (self.names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=2)

            # t1 = time_synchronized()
            t2 = time_synchronized()
            print('Time Taken: ', t2 - t1, "\t Total Objects: ", total_objects)

            if view_img:
                cv2.imshow("Image", im0)
                code = cv2.waitKeyEx(1)
                if code == 113:
                    rospy.signal_shutdown("Kill Requested")
                    exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='../../../../../../../Storage/Other/Temp/dataset/sequences/00/image_0',help='source')  # file/folder, 0 for webcam
    parser.add_argument('--lidar', type=str, default='../../../../../../../Storage/Other/Temp/dataset/velodyne_dataset/sequences/00/velodyne/', help='lidar')

    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default=True, action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    try:
        rospy.init_node("SemanticBoxes", anonymous=True)
        app = YOLOv5()
        rate = rospy.Rate(10) # 10hz
        while not rospy.is_shutdown():
            rate.sleep()

    except rospy.ROSInterruptException:
        pass

