from detector import *
from renderer import *

class SemanticSLAMApp:

    def __init__(self):
        self.renderer = Renderer(render=True, show_origin=True)

    def display_image(self, tag, img, delay):
        cv2.imshow(tag, img)
        code = cv2.waitKeyEx(delay)
        return code

    def dataset_main(self):
        home = os.path.expanduser('~')
        path = home + '/.ihmc/logs/perception/'
        filename = 'KITTI_Dataset_00.hdf5'
        group = '/kitti/left/'
        weights_file = "./yolov8n-seg.pt"

        detector = Detector(weights_file)

        data = h5py.File(path + filename, 'r')


        pose = np.eye(4)

        for index in range(len(data[group].keys())):

            pose[:,:3] += np.array([0.01, 0, 0])

            self.renderer.submit_pose(pose)

            self.renderer.update()

            print(data[group + str(index)])

            buffer = data[group + str(index)][:].view('uint8')
            buffer_image = np.asarray(buffer, dtype=np.uint8)
            buffer_image = cv2.imdecode(buffer_image, cv2.IMREAD_GRAYSCALE)
            buffer_image = cv2.cvtColor(buffer_image, cv2.COLOR_GRAY2RGB)

            result = detector.detect(buffer_image)[0]

            image = result.plot()

            cv2.imshow("Frame", image)
            code = cv2.waitKeyEx(30)

            print(code)

            if code == 1048689 or code == 113:
                data.close()
                break
        
        data.close()


if __name__ == "__main__":
    app = SemanticSLAMApp()
    app.dataset_main()



