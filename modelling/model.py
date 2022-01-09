import cv2
import os
import numpy as np

files1 = os.listdir('../../Data/cam1/')
files2 = os.listdir('../../Data/cam2/')

n_files1 = len(files1)
n_files2 = len(files2)

first1 = cv2.imread("../../Data/cam1/img1_" + str(1) + ".png", cv2.IMREAD_COLOR)
first2 = cv2.imread("../../Data/cam2/img2_" + str(1) + ".png", cv2.IMREAD_COLOR)

for num_frames in range(20, n_files1):
    img1 = cv2.imread("../../Data/cam1/img1_" + str(num_frames) + ".png", cv2.IMREAD_COLOR)
    img2 = cv2.imread("../../Data/cam2/img2_" + str(num_frames) + ".png", cv2.IMREAD_COLOR)

    diff1 = cv2.absdiff(img1, first1)
    mask1 = cv2.cvtColor(diff1, cv2.COLOR_BGR2GRAY)
    thresh1 = 45
    imask1 = mask1 > thresh1
    canvas1 = np.zeros_like(img1, np.uint8)
    canvas1[imask1] = img1[imask1]

    diff2 = cv2.absdiff(img2, first2)
    mask2 = cv2.cvtColor(diff2, cv2.COLOR_BGR2GRAY)
    thresh2 = 15
    imask2 = mask2 > thresh2
    canvas2 = np.zeros_like(img2, np.uint8)
    canvas2[imask2] = img2[imask2]

    tot1, tot2 = np.sum(imask1), np.sum(imask2)

    final = cv2.hconcat([canvas1, canvas2])
    cv2.imshow("Final", final)
    print(num_frames)
    # cv2.namedWindow('Result1', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Result1', 800, 600)
    # cv2.imshow("Result1", canvas1)
    #
    # cv2.namedWindow('Result2', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Result2', 800, 600)
    # cv2.imshow("Result2", canvas2)

    code = cv2.waitKeyEx(600)
    if code == ord('q'):
        break

cv2.destroyAllWindows()
