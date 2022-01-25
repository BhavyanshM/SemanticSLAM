import cv2
import os


img_path = '/home/quantum/Workspace/Storage/Other/Temp/dataset/DoorDetect-Dataset/images/'
label_path = '/home/quantum/Workspace/Storage/Other/Temp/dataset/DoorDetect-Dataset/labels/'


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

classes = ['door', 'handle', 'cabinet door', 'refrigerator door']


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
    count = 0
    for file in img_files:

        print('Opening(', count, '): ', label_path + file[:-4])
        img = cv2.imread(img_path + file)
        labels = open(label_path + '.'.join(file.split('.')[:-1]) + '.txt')

        for l in labels:
            det = list(map(float, l.split(' ')))
            x, y, w, h = int(det[1] * img.shape[1]), int(det[2] * img.shape[0]), int(det[3] * img.shape[1]), int(det[4] * img.shape[0])
            # print(x,y,w,h)
            print(classes[int(det[0])], int(det[0]), x, y, w, h)

            cv2.rectangle(img, (x - int(w/2), y - int(h/2)), (x + int(w/2), y + int(h/2)), (24, 244, 100), 3)

        # print(file)
        cv2.imshow("Image", img)
        code = cv2.waitKeyEx(1)
        count += 1
        if code == 113:
            exit()

visualize(img_files, img_path, label_files, label_path)