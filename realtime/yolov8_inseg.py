from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2
import h5py
import os
import numpy as np

def set_camera_props(cap):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)

def run_camera():
    while True:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def display_image(tag, img, delay):
    cv2.imshow(tag, img)
    code = cv2.waitKeyEx(delay)
    return code


def camera_main():
    cap = cv2.VideoCapture(0)
    # set_camera_props(cap)

    model = YOLO("./yolov8n-seg.pt")

    scale = 2.0

    while True:
        ret, frame = cap.read()


        result = model.predict(source=frame, show=False, conf=0.5)

        result_frame = result[0].plot()


    

        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Frame", int(result_frame.shape[1] * scale), int(result_frame.shape[0] * scale))
        code = display_image("Frame", result_frame, 1)

        code = cv2.waitKeyEx(1)
        if code == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def dataset_main():
    home = os.path.expanduser('~')
    path = home + '/.ihmc/logs/perception/'
    weights_path = home + '/.ihmc/weights/'
    filename = 'KITTI_Dataset_00.hdf5'
    group = '/kitti/left/'
    # weights_file = "./yolov8n-seg.pt"
    weights_file = "facade_yolov8n_100.pt"

    model = YOLO(weights_path + weights_file)

    data = h5py.File(path + filename, 'r')


    for index in range(len(data[group].keys())):

        print(data[group + str(index)])

        buffer = data[group + str(index)][:].view('uint8')
        buffer_image = np.asarray(buffer, dtype=np.uint8)
        buffer_image = cv2.imdecode(buffer_image, cv2.IMREAD_GRAYSCALE)
        buffer_image = cv2.cvtColor(buffer_image, cv2.COLOR_GRAY2RGB)

        result = model.predict(source=buffer_image, show=True, conf=0.5)

        # cv2.imshow("Frame", buffer_image)
        code = cv2.waitKeyEx(30)

        if code == 1048689:
            data.close()
            break
    
    data.close()


if __name__ == "__main__":
    dataset_main()
    # camera_main()


