import numpy as np
import cv2
from utils import *

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
        self.params = {'MIN_DISPARITY': 5, 'NUM_DISPARITIES': 100, 'BLOCK_SIZE': 3, 'DISP_12_MAX_DIFF': 10, 'UNIQUENESS_RATIO': 0, 'SPECKLE_RANGE': 0, 'SPECKLE_WIN_SIZE': 0, 'MAX_DISPARITY': 83}

        self.disparity = None

    def show_trackbars(self):
        cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
        cv2.createTrackbar("MIN_DISPARITY", "Controls", self.params['MIN_DISPARITY'], 100, lambda x: self.on_change(x, "MIN_DISPARITY"))
        cv2.createTrackbar("NUM_DISPARITIES", "Controls", self.params['NUM_DISPARITIES'], 100, lambda x: self.on_change(x, "NUM_DISPARITIES"))
        cv2.createTrackbar("MAX_DISPARITY", "Controls", self.params['MAX_DISPARITY'], 100, lambda x: self.on_change(x, "MAX_DISPARITY"))
        cv2.createTrackbar("BLOCK_SIZE", "Controls", self.params['BLOCK_SIZE'], 100, lambda x: self.on_change(x, "BLOCK_SIZE"))
        cv2.createTrackbar("UNIQUENESS_RATIO", "Controls", self.params['UNIQUENESS_RATIO'], 100, lambda x: self.on_change(x, "UNIQUENESS_RATIO"))
        cv2.createTrackbar("SPECKLE_WIN_SIZE", "Controls", self.params['SPECKLE_WIN_SIZE'], 100, lambda x: self.on_change(x, "SPECKLE_WIN_SIZE"))
        cv2.createTrackbar("SPECKLE_RANGE", "Controls", self.params['SPECKLE_RANGE'], 100, lambda x: self.on_change(x, "SPECKLE_RANGE"))
        cv2.createTrackbar("DISP_12_MAX_DIFF", "Controls", self.params['DISP_12_MAX_DIFF'], 100, lambda x: self.on_change(x, "DISP_12_MAX_DIFF"))

    def on_change(self, input, name):
        self.params[name] = input
        self.stereo.setMinDisparity(self.params["MIN_DISPARITY"])
        self.stereo.setNumDisparities(self.params["NUM_DISPARITIES"])
        self.stereo.setBlockSize(self.params["BLOCK_SIZE"])
        self.stereo.setDisp12MaxDiff(self.params["DISP_12_MAX_DIFF"])
        self.stereo.setUniquenessRatio(self.params["UNIQUENESS_RATIO"])
        self.stereo.setSpeckleRange(self.params["SPECKLE_RANGE"])
        self.stereo.setSpeckleWindowSize(self.params["SPECKLE_WIN_SIZE"])
        print(self.params)


    def compute_stereo_depth(self, left, right):

        self.disparity = self.stereo.compute(left, right)
        depthMap = 718 * 0.54 / np.abs(self.disparity)
        depth = np.array(depthMap, dtype=np.uint16)

        disparity = self.disparity.astype(np.float32)
        disparity = (disparity / 16.0 - self.min_disp) / self.num_disp

        norm_image = cv2.normalize(disparity, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        return depthMap