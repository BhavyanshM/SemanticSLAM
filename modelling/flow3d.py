import cv2

cap = cv2.VideoCapture(2, cv2.CAP_V4L2)


def set_cam_props(cam):
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 848)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv2.CAP_PROP_FPS, 30)
    # cam.set(cv2.CAP_PROP_MODE, cv2.CAP_OPENCV_MJPEG)
    cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cam.set(cv2.CAP_PROP_EXPOSURE, -4)
    return cam

cap = set_cam_props(cap)

while True:
    ret, frame = cap.read()

    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
