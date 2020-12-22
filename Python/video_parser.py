import cv2
import argparse
import os
import sys


def write_video(dir, filename):

    files = sorted(os.listdir(dir))
    print(dir + files[0])
    frame = cv2.imread(dir + files[0], cv2.IMREAD_COLOR)


    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    out = cv2.VideoWriter(filename, fourcc=fourcc, fps=20.0, frameSize=(frame.shape[1], frame.shape[0]))

    for f in files:
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)
        print(f)
        frame = cv2.imread(dir + f, cv2.IMREAD_COLOR)
        out.write(frame)

    out.release()


def read_video(filename):
    print(filename)
    cap = cv2.VideoCapture(filename)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)
        else:
            print("Done")
            break
    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='Directory of Images')
    opt = parser.parse_args()

    # write_video(opt.dir, "../videos/tum_rgb_xyz.avi")

    read_video("../videos/tum_rgb_xyz.avi")


