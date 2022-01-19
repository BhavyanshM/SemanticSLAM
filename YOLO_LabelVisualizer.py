import cv2
import os


img_path = '/home/quantum/Workspace/Storage/Other/Temp/dataset/YOLO_Data/data/images/train2017/'
label_path = '/home/quantum/Workspace/Storage/Other/Temp/dataset/YOLO_Data/data/labels/train2017/'
archive_path = '/home/quantum/Workspace/Storage/Other/Temp/dataset/YOLO_Data/data/archive/'
facade_img_path = '/home/quantum/Workspace/Storage/Other/Temp/dataset/YOLO_Data/data/images/facade/'
facade_label_path = '/home/quantum/Workspace/Storage/Other/Temp/dataset/YOLO_Data/data/labels/facade/'

# ---------------------For Visualizing COCO dataset ------------------------------------------------
img_files = os.listdir(img_path)
label_files = os.listdir(label_path)

# -------------------- For visualizing FacadeLabelMe dataset ---------------------------------------
# facade_img_files = sorted(os.listdir(facade_img_path))
# facade_label_files = sorted(os.listdir(facade_label_path))
# img_files = facade_img_files
# label_files = facade_label_files
# img_path = facade_img_path
# label_path = facade_label_path

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
        'teddy bear', 'hair drier', 'toothbrush']

classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
           'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
           'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
           'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
           'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
           'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush', 'window']

print('Total Classes: ', len(classes))

def sync_labels():
    count = 0
    for x in label_files:
        if os.path.exists(img_path + x[:-4] + '.jpg'):
            print('Found: ', x[:-4] + '.jpg')
        else:
            count += 1
            # os.rename(facade_img_path + x, archive_path + x)
            print('Not Found : ', x[:-4] + '.jpg', count)

def visualize(img_files, img_path, label_files, label_path):
    for file in img_files:
        img = cv2.imread(img_path + file)
        labels = open(label_path + file[:-4] + '.txt')

        for l in labels:
            det = list(map(float, l.split(' ')))
            x, y, w, h = int(det[1] * img.shape[1]), int(det[2] * img.shape[0]), int(det[3] * img.shape[1]), int(det[4] * img.shape[0])
            # print(x,y,w,h)
            print(classes[int(det[0])], int(det[0]), x, y, w, h)

            cv2.rectangle(img, (x - int(w/2), y - int(h/2)), (x + int(w/2), y + int(h/2)), (24, 244, 100), 3)

        # print(file)
        cv2.imshow("Image", img)
        code = cv2.waitKeyEx(0)
        if code == 113:
            exit()

visualize(img_files, img_path, label_files, label_path)