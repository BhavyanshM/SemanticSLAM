import cv2
from SemanticFeature import *

def create_tracks(detections):
    objects = []
    for i, object in enumerate(detections):
        instance = SemanticFeature(object[0], i, np.array([object[1], object[2]]), 0, None, object)
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

def plot_detection_boxes(img, objects, thickness=2, names=[]):
    for det in objects:
        cls, x, y, w, h = tuple(det.bbox)
        color = (123 * cls % 255, 231 * cls % 255, 314 * cls % 255)
        color_text = (110 * cls % 255, 83 * cls % 255, 121 * cls % 255)
        cv2.rectangle(img, (int(x-w/2),int(y-h/2)), (int(x+w/2),int(y+h/2)), color, thickness)
        if len(names) > 0:
            cv2.putText(img, names[cls], (int(x-w/2), int(y-h/2 - 2)), 0, 0.6, color_text, thickness=2, lineType=cv2.LINE_AA)

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
    cv2.resizeWindow("Image", int(img.shape[1]*1.3), int(img.shape[0]*1.3))
    cv2.imshow("Image", img)
    print("Shape:", img.shape)
    code = cv2.waitKeyEx(20)
    if code == 32:
        code = cv2.waitKeyEx(0)
    if code == 113:
        exit()

def combine_images_vertical(img1, img2):
    images = [img1, img2]
    final = cv2.vconcat(images)
    return final

def plot_associations(img, mot, detections):
    for i in range(len(mot.features)):
        if mot.table[i, mot.matches[i]] != 0:
            cls = mot.features[i].cls
            color = (123 * cls % 255, 231 * cls % 255, 314 * cls % 255)
            cv2.line(img, (mot.features[i].bbox[1], mot.features[i].bbox[2]), (detections[mot.matches[i]].bbox[1], int(img.shape[0]/2) + detections[mot.matches[i]].bbox[2]), color, 3)

