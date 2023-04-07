import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from semantic_feature import *

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

def display(img, name='Image', delay=0):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, int(img.shape[1]*1.3), int(img.shape[0]*1.3))
    cv2.imshow(name, img)
    code = cv2.waitKeyEx(delay)
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

def get_plane(point, normal):
    plane = np.array([normal[0], normal[1], normal[2], -np.dot(point, normal)])
    return plane

def get_plane_z(p, pi):
    z = -(np.dot(pi[:2], p) + pi[3]) / pi[2]
    return z

def find_plane_intersection(pi1, pi2, pi3):

    print("Intersection:", pi1, pi2, pi3)

    A = np.zeros(shape=(3, 4))
    A[0, :] = pi1
    A[1, :] = pi2
    A[2, :] = pi3

    # A_inv = np.linalg.inv(A.T @ A) @ A.T
    U, S, V = np.linalg.svd(A.T)

    x = U[:, 3]
    x /= x[3]

    return x[:3]

def get_object_location_from_size(det, size, focal_length=718):
    fx, cx, fy, cy = 718, 607, 718, 185
    cls, x, y, w, h = det
    depth = focal_length * size / det[3]
    point = np.array([(x - cx) / fx, (y - cy) / fy, 1]) * depth
    return point


def triangulate_convex_polytope(det1, det2, pose, axis=0):

    ps1 = get_planes_from_detection(det1, axis)
    ps2 = get_planes_from_detection(det2, axis)

    pi_const_1 = get_plane(np.array([axis,1-axis,0]), np.array([axis,1-axis,0]))
    pi_const_2 = get_plane(np.array([-axis, -1+axis, 0]), np.array([axis, 1-axis, 0]))

    tp1, np1 = transform_plane(np.hstack([np.array([0, 0, 0]), ps2[0][:3]]), pose)
    tp2, np2 = transform_plane(np.hstack([np.array([0, 0, 0]), ps2[1][:3]]), pose)

    pi_ps2 = (np.hstack([tp1, np1[:3]]), np.hstack([tp2, np2[:3]]))

    ps2 = (np1, np2)

    points = []
    points.append(find_plane_intersection(ps1[0], ps2[0], pi_const_1))
    points.append(find_plane_intersection(ps1[0], ps2[1], pi_const_1))
    points.append(find_plane_intersection(ps1[1], ps2[1], pi_const_1))
    points.append(find_plane_intersection(ps1[1], ps2[0], pi_const_1))

    points.append(find_plane_intersection(ps1[0], ps2[0], pi_const_2))
    points.append(find_plane_intersection(ps1[0], ps2[1], pi_const_2))
    points.append(find_plane_intersection(ps1[1], ps2[1], pi_const_2))
    points.append(find_plane_intersection(ps1[1], ps2[0], pi_const_2))

    return ps1, pi_ps2, np.vstack(points)



def get_planes_from_detection(det, axis=0):
    fx, cx, fy, cy = 718, 607, 718, 185
    cls, x, y, w, h = det

    vec_tl = np.array([((x - w/2) - cx) / fx, ((y - h/2) - cy) / fy, 1])
    vec_bl = np.array([((x - w / 2) - cx) / fx, ((y + h / 2) - cy) / fy, 1])
    vec_tr = np.array([((x + w / 2) - cx) / fx, ((y - h / 2) - cy) / fy, 1])
    vec_br = np.array([((x + w / 2) - cx) / fx, ((y + h / 2) - cy) / fy, 1])

    if axis == 0:
        normal_l = np.cross(vec_tl, vec_bl)
        normal_r = np.cross(vec_tr, vec_br)
    elif axis == 1:
        normal_l = np.cross(vec_tl, vec_tr)
        normal_r = np.cross(vec_bl, vec_br)

    pi_l = get_plane(np.array([0, 0, 0]), normal_l)
    pi_r = get_plane(np.array([0, 0, 0]), normal_r)

    return [pi_l, pi_r]


def transform_plane(plane, transform):
    p = np.ones(shape=(4,))
    p[:3] = plane[:3]

    point = (transform @ p)[:3]
    normal = transform[:3,:3] @ plane[3:]

    pi = get_plane(point, normal)

    return (point, pi)

def create_pointcloud_from_depth(depth):
    fx, cx, fy, cy = 718, 607, 718, 185
    points = []
    for y in range(depth.shape[0]):
        for x in range(depth.shape[1]):
            Z = depth[y,x]
            Y = (y - cy) / fy * Z
            X = (x - cx) / fx * Z
            points.append(np.array([X, Y, Z]))
    points = np.array(points)
    return points[points[:,2] < 200]

def extract_object_points(depth, objects):
    fx, cx, fy, cy = 718, 607, 718, 185
    clouds = []
    X = np.zeros_like(depth)
    Y = np.zeros_like(depth)

    for i in range(X.shape[1]):
        X[:,i] = i
        print(X[:,i].tolist())

    for i in range(Y.shape[0]):
        Y[i,:] = i

    print(Y)

    size = depth.shape[0] * depth.shape[1]
    X = X.reshape((size,))
    Y = Y.reshape((size,))
    D = depth.reshape((size, ))

    cloud = np.vstack([X,Y,D])


    for obj in objects:
        cls, x, y, w, h = obj.bbox
        condition_x = np.logical_and(cloud[0,:] > x - w/2, cloud[0, :] < x + w / 2)
        condition_y = np.logical_and(cloud[1, :] > y - h / 2, cloud[1, :] < y + h / 2)
        condition_z = np.logical_and(cloud[2, :] > 0.1, cloud[2, :] < 1000)
        condition = np.logical_and(condition_x, condition_y, condition_z)
        points = cloud[:, condition]

        xs, ys, zs = points[0, :], points[1, :], points[2, :]

        points[0, :] = (xs - cx) / fx * zs
        points[1, :] = (ys - cy) / fy * zs

        clouds.append(points.T)

        print(cls, points.shape, x, y, w, h)

    return clouds


def render_slam(poses, landmarks, matches, renderer):
    for pose in poses:
        renderer.submit_sphere(pose, radius=0.1, color=[0.3, 0.8, 0.5])

    for l in landmarks:
        renderer.submit_sphere(l, radius=0.04, color=[0.8, 0.4, 0.9])

    # for pose in poses:
    #     for l in landmarks:

    graph_points = np.vstack([poses, landmarks])
    print(graph_points.shape)

    line_indices = []
    line_colors = []
    for i in range(poses.shape[0] - 1):
        line_indices.append([i, i + 1])
        line_colors.append([0.2, 0.7, 0.4])

    for m in matches:
        i, j = m
        line_indices.append([i, poses.shape[0] + j])
        line_colors.append([0.1, 0.3, 0.4])

    renderer.submit_lines(graph_points, np.array(line_indices), np.array(line_colors))


def plot_sparse_matrix(A):
    plt.figure(figsize=(10,10))
    plt.spy(A)
    plt.show()


def compute_winding_number(point, hull):
    total_angle = 0
    for i in range(len(hull) - 1):
        v1, v2 = hull[i] - point, hull[i+1] - point
        cosim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cross = np.cross(v1, v2)
        angle = math.acos(cosim) * cross / math.fabs(cross)
        total_angle += angle
    return total_angle












