import cv2
import pixellib
from pixellib.torchbackend.instance import instanceSegmentation


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

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    set_camera_props(cap)

    segment_video = instanceSegmentation()
    segment_video.load_model("./pointrend_resnet50.pkl", confidence=0.7, detection_speed="fast")
    segment_video.process_camera(cap, show_bboxes=True, frames_per_second=30, check_fps=True, show_frames=True, frame_name="Frame")




