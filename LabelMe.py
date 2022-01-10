import torch
import cv2
import os
import numpy as np

various = torch.Tensor([0,0,0]).__reversed__()
building = torch.Tensor([128,0,0]).__reversed__()
car = torch.Tensor([128,0,128]).__reversed__()
door = torch.Tensor([128,128,0]).__reversed__()
pavement = torch.Tensor([128,128,128]).__reversed__()
road = torch.Tensor([128,64,0]).__reversed__()
sky = torch.Tensor([0,128,128]).__reversed__()
vegetation = torch.Tensor([0,128,0]).__reversed__()
window = torch.Tensor([0,0,128]).__reversed__()

path = '/home/quantum/Workspace/Storage/Other/Temp/dataset/YOLO_Data/data/images/facade/'
facade_labels_path = '/home/quantum/Workspace/Storage/Other/Temp/dataset/YOLO_Data/data/labels/facade/'
annotations_path = '/home/quantum/Workspace/Storage/Other/Temp/dataset/labelmefacade/labels/'
files = sorted(os.listdir(path))
label_files = sorted(os.listdir(annotations_path))

def write_box(f, rect, height, width):
    x, y, w, h = rect
    f.write("91 {} {} {} {}\n".format((x + w/2) / width, (y + h/2) / height, (w / width), (h / height)))

start = 0

for i in range(start, len(files)):




    print(i, "Reading: ", files[i], label_files[i], end='')

    if files[i][-4:] != '.jpg':
        continue

    f = open(facade_labels_path + files[i][:-4] + '.txt', '+w')


    img = cv2.imread(path + files[i])
    label = cv2.imread(annotations_path + label_files[i])

    img_tensor = torch.Tensor(img)
    mask = torch.Tensor(label)

    # split the color-encoded mask into a set of boolean masks.
    # Note that this snippet would work as well if the masks were float values instead of ints.
    masks = torch.all(mask == window, dim=2)

    masks = np.array(masks, dtype=np.int8) * 120

    n_labels, labels = cv2.connectedComponents(masks)

    # +++++++++++++++++++++++++++++++++ Visualization +++++++++++++++++++++++++++++++++++
    # Map component labels to hue val, 0-179 is the hue range in OpenCV

    for k in range(1, n_labels):

        detection = np.array(labels == k, dtype=np.uint8)

        contours, _ = cv2.findContours(detection, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        polygon = cv2.approxPolyDP(contours[0], 3, True)
        boundRect = cv2.boundingRect(polygon)


        write_box(f, boundRect, img.shape[0], img.shape[1])

        cv2.rectangle(img, (int(boundRect[0]), int(boundRect[1])),(int(boundRect[0] + boundRect[2]), int(boundRect[1] + boundRect[3])),
                      (255,0,255), 2)



        label_hue = np.uint8(179 * detection / np.max(detection))
        blank_ch = 255 * np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

        # Converting cvt to BGR
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

        # set bg label to black
        labeled_img[label_hue == 0] = 0

        # Showing Original Image
        # cv2.imshow("Windows", cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    print(img.shape, label.shape, end='')
    cv2.imshow("Final", img)
    cv2.imshow("Label", label)
    code = cv2.waitKeyEx(1)

    if code == 113:
        f.close()
        exit()

    print()
    f.close()

