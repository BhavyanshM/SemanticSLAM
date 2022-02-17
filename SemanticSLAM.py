import numpy as np
import cv2

class SemanticSLAM:
    def __init__(self):
        self.win_size = 5
        self.min_disp = -1
        self.max_disp = 63  # min_disp * 9
        self.num_disp = self.max_disp - self.min_disp  # Needs to be divisible by 16
        self.stereo = cv2.StereoSGBM_create(minDisparity=self.min_disp,
                                            numDisparities=self.num_disp,
                                            blockSize=5,
                                            uniquenessRatio=5,
                                            speckleWindowSize=5,
                                            speckleRange=5,
                                            disp12MaxDiff=1,
                                            P1=8 * 3 * self.win_size ** 2,  # 8*3*win_size**2,
                                            P2=32 * 3 * self.win_size ** 2)  # 32*3*win_size**2)
        self.disparity = None


    def compute_stereo_depth(self, left, right):
        self.disparity = self.stereo.compute(left, right)
        depthMap = 65536 / np.abs(self.disparity)
        depth = np.array(depthMap, dtype=np.uint16)

        disparity = self.disparity.astype(np.float32)
        disparity = (disparity / 16.0 - self.min_disp) / self.num_disp

        norm_image = cv2.normalize(disparity, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        print(norm_image)

        cv2.imshow("Final", norm_image)
        cv2.waitKey(0)

        return disparity