import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized


def init_calib():
    kitti_data = open("../Python/calib/kitti_calib.txt").readlines()
    calibs = {line.split(":")[0]: line.split(":")[1].strip('\n') for line in kitti_data}
    P2 = np.matrix([float(x) for x in calibs["P2"].split(' ')[1:]]).reshape(3, 4)
    R0_rect = np.matrix([float(x) for x in calibs["R0_rect"].split(' ')[1:]]).reshape(3, 3)
    R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0], axis=0)
    R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0, 1], axis=1)
    Tr_velo_to_cam = np.matrix([float(x) for x in calibs["Tr_velo_to_cam"].split(' ')[1:]]).reshape(3, 4)
    Tr_velo_to_cam = np.insert(Tr_velo_to_cam, 3, values=[0, 0, 0, 1], axis=0)
    return P2, R0_rect, Tr_velo_to_cam


def get_lidar_on_img(file, P2, R0_rect, Tr_velo_to_cam, width, height):
    scan = np.fromfile(file, dtype=np.float32)
    scan = np.array(scan, dtype=np.float64)
    scan = scan.reshape((-1, 4))
    xyz = scan[:, :3]
    xyz = xyz[::10]

    vel_points = np.insert(xyz, 3, 1, axis=1).T
    vel_points = np.delete(vel_points, np.where(vel_points[0, :] < 0), axis=1)
    cam_points = P2 * R0_rect * Tr_velo_to_cam * vel_points
    cam_points = np.delete(cam_points, np.where(cam_points[2, :] < 0)[1], axis=1)
    cam_points[:2] /= cam_points[2, :]

    u, v, z = cam_points
    u_out = np.logical_or(u < 0, u > width)
    v_out = np.logical_or(v < 0, v > height)
    outlier = np.logical_or(u_out, v_out)
    cam_points = np.delete(cam_points, np.where(outlier), axis=1)

    return xyz, cam_points


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    P2, R0_rect, Tr_velo_to_cam = init_calib()
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    rend_opt = vis.get_render_option()
    rend_opt.background_color = np.asarray([0, 0, 0])
    pcd = o3d.geometry.PointCloud()
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=(0, 0, 0))

    dir = opt.lidar
    lidar = os.listdir(dir)
    lidar = sorted(lidar)
    file = dir + lidar[0]
    scan = np.fromfile(file, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    xyz = scan[:, :3]
    pcd.points = o3d.utility.Vector3dVector(xyz)
    vis.add_geometry(pcd)
    vis.add_geometry(axes)

    f = open('labels.txt', 'w+')
    time_count = 0
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:

        t1 = time_synchronized()
        file = dir + lidar[time_count]
        time_count += 1

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=opt.augment)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = Path(path[i]), '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = Path(path), '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = map(int, (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist())  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

        # t1 = time_synchronized()
        xyz, cam_points = get_lidar_on_img(file, P2, R0_rect, Tr_velo_to_cam, im0.shape[1], im0.shape[0])
        pcd.points = o3d.utility.Vector3dVector(xyz)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        u, v, z = cam_points
        # for i in range(u.shape[1]):
        #     cv2.circle(im0, (int(u[0, i]), int(v[0, i])), 2, (int(z[0, i] / 20 * 255), 200, int(z[0, i] / 20 * 255)),-1)

        t2 = time_synchronized()
        print('%sDone. (%.3fs)' % (s, t2 - t1))

        if view_img:
            # im0 = cv2.resize(im0, (im0.shape[1], im0.shape[0]))
            cv2.imshow("Image", im0)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration

    print('Done. (%.3fs)' % (time.time() - t0))
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='../../Data_Links/kitti/00/image_0/',
                        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--lidar', type=str, default='../../Data_Links/kitti_velodyne/00/velodyne/', help='lidar')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
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

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
