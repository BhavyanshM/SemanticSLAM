import cv2

def set_camera_props(cap):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)

def run_camera(cap):
    while True:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def camera_main():
    cap = cv2.VideoCapture(0)
    self.set_camera_props(cap)

    model = YOLO("./yolov8n-seg.pt")

    while True:
        ret, frame = cap.read()

        result = model.predict(source=frame, show=True, conf=0.5)

        print(result.boxes)

        # code = display_image("Frame", frame, 1)

        code = cv2.waitKeyEx(1)
        if code == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()