import cv2
from Instance import *

def create_instances(detections):
    objects = []
    for i, object in enumerate(detections):
        instance = Instance(object[0], i, np.array([object[1], object[2]]), 0, None, object)
        objects.append(instance)

    return objects


def load_detections(file):
    f = open(file)
    objects = []
    for l in f:
        det = list(map(int, l.split(' ')))
        objects.append(det)
    return objects

def print_detection_classes(detections, classes):
    s = ''
    for det in detections:
        if classes[det[0]] not in s:
            s += ' ' + classes[det[0]]
    print('Classes:', s)

def plot_detection_boxes(img, objects, thickness=2):
    for det in objects:
        cls, x, y, w, h = tuple(det.bbox)
        cv2.rectangle(img, (int(x-w/2),int(y-h/2)), (int(x+w/2),int(y+h/2)), (123*cls % 255, 231*cls % 255, 314*cls % 255), thickness)

def xywh_to_xyxy(xywh):
    return [int(xywh[0] - xywh[2]/2), int(xywh[1] - xywh[3]/2), int(xywh[0] + xywh[2]/2), int(xywh[1] + xywh[3]/2)]

def compute_iou(bbox1, bbox2):
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    int_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    iou = int_area / float(bbox1_area + bbox2_area - int_area)

    return iou

def display(img):
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", img.shape[1]*2, img.shape[0]*2)
    cv2.imshow("Image", img)
    code = cv2.waitKeyEx(50)
    if code == 32:
        code = cv2.waitKeyEx(0)
    if code == 113:
        exit()

def combine_images_vertical(img1, img2):
    images = [img1, img2]
    final = cv2.vconcat(images)
    return final
