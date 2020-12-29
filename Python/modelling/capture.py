import cv2
import numpy as np
import time

cap1 = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap2 = cv2.VideoCapture(2, cv2.CAP_V4L2)


def set_cam_props(cam):
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 848)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv2.CAP_PROP_FPS, 30)
    # cam.set(cv2.CAP_PROP_MODE, cv2.CAP_OPENCV_MJPEG)
    cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cam.set(cv2.CAP_PROP_EXPOSURE, -4)
    return cam


cap1 = set_cam_props(cap1)
cap2 = set_cam_props(cap2)

# ret, prev = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
prevTime = time.time()
num_frames = 0
total_time = 0

# time.sleep(5)
ret1, first1 = cap1.read()
ret2, first2 = cap2.read()
set1, set2 = 0, 0
print("Captured!")

while True:
    num_frames += 1
    ret1, img1 = cap1.read()
    ret2, img2 = cap2.read()

    cv2.namedWindow('Result1', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Result1', 800, 600)
    cv2.imshow("Result1", img1)

    cv2.namedWindow('Result2', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Result2', 800, 600)
    cv2.imshow("Result2", img2)

    cv2.imwrite("../../Data/cam1/img1_" + str(num_frames) + ".png", img1)
    cv2.imwrite("../../Data/cam2/img2_" + str(num_frames) + ".png", img2)

    code = cv2.waitKeyEx(1)
    if code == ord('q'):
        break

cv2.destroyAllWindows()
