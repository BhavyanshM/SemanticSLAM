import numpy as np
import cv2
import scipy.sparse.linalg
from scipy import sparse
from utils import *
import sys

np.set_printoptions(threshold=sys.maxsize)


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
        self.params = {'MIN_DISPARITY': 5, 'NUM_DISPARITIES': 100, 'BLOCK_SIZE': 3, 'DISP_12_MAX_DIFF': 10,
                       'UNIQUENESS_RATIO': 0, 'SPECKLE_RANGE': 0, 'SPECKLE_WIN_SIZE': 0, 'MAX_DISPARITY': 83}

        self.disparity = None

    def show_trackbars(self):
        cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
        cv2.createTrackbar("MIN_DISPARITY", "Controls", self.params['MIN_DISPARITY'], 100,
                           lambda x: self.on_change(x, "MIN_DISPARITY"))
        cv2.createTrackbar("NUM_DISPARITIES", "Controls", self.params['NUM_DISPARITIES'], 100,
                           lambda x: self.on_change(x, "NUM_DISPARITIES"))
        cv2.createTrackbar("MAX_DISPARITY", "Controls", self.params['MAX_DISPARITY'], 100,
                           lambda x: self.on_change(x, "MAX_DISPARITY"))
        cv2.createTrackbar("BLOCK_SIZE", "Controls", self.params['BLOCK_SIZE'], 100,
                           lambda x: self.on_change(x, "BLOCK_SIZE"))
        cv2.createTrackbar("UNIQUENESS_RATIO", "Controls", self.params['UNIQUENESS_RATIO'], 100,
                           lambda x: self.on_change(x, "UNIQUENESS_RATIO"))
        cv2.createTrackbar("SPECKLE_WIN_SIZE", "Controls", self.params['SPECKLE_WIN_SIZE'], 100,
                           lambda x: self.on_change(x, "SPECKLE_WIN_SIZE"))
        cv2.createTrackbar("SPECKLE_RANGE", "Controls", self.params['SPECKLE_RANGE'], 100,
                           lambda x: self.on_change(x, "SPECKLE_RANGE"))
        cv2.createTrackbar("DISP_12_MAX_DIFF", "Controls", self.params['DISP_12_MAX_DIFF'], 100,
                           lambda x: self.on_change(x, "DISP_12_MAX_DIFF"))

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

    def generate_sample_slam(self):
        t = np.linspace(0, 2 * np.pi, 10)
        y, x, z = np.zeros_like(t), 2 * np.sin(t), t * 2
        poses = np.vstack([x, y, z]).T

        # n_landmarks = 12
        # landmarks = np.vstack([np.random.uniform(-4, 4, n_landmarks), np.random.uniform(2, -2, n_landmarks),
        #                        np.random.uniform(0, 15, n_landmarks)]).T
        # print(landmarks.tolist())

        landmarks = np.array([[-2.256757860111996, -1.978569124634991, 2.540965322679638],
                              [3.255220228187416, -1.6231162856471246, 0.2021717433538428],
                              [-1.9255540176639805, -1.426716846561479, 6.268101750239819],
                              [0.24298209883978483, 0.342804505449422, 8.59584405722175],
                              [2.9875926564258766, 0.12188092419101393, 14.129723080181174],
                              [-0.4577048372804473, -1.6932602092219922, 2.5022351660686324],
                              [-1.2807591606494384, -0.35284340056709285, 9.476789150706205],
                              [2.8434652692648337, -0.10738923235083853, 4.425715535465578],
                              [-3.4548462301495793, 1.963896200828168, 6.804982442987478],
                              [2.2909538820587967, -1.4579195354305456, 12.031965371931667],
                              [0.26895515542166404, 1.4470594708003786, 12.3019930044398],
                              [1.7048254379979708, -1.3525032472937606, 0.4689799165503816]])

        associations = []
        max_dist = 5

        for i in range(poses.shape[0]):
            for j in range(landmarks.shape[0]):
                dist = np.linalg.norm(poses[i] - landmarks[j])
                if dist < max_dist:
                    associations.append((i, j))

        measurements = np.zeros(shape=(poses.shape[0] - 1 + len(associations), 3))

        for i, m in enumerate(associations):
            measurements[poses.shape[0] - 1 + i, :] = landmarks[m[1]] - poses[m[0]]

        for i in range(poses.shape[0] - 1):
            measurements[i] = poses[i + 1] - poses[i]

        noise = np.random.multivariate_normal(mean=np.array([0, 0, 0]), cov=np.eye(3), size=measurements.shape[0])
        measurements += noise

        return poses, landmarks, associations, measurements

    def solve_positions(self, measurements, associations, num_poses, num_landmarks):
        print(measurements.shape)

        A = np.zeros(shape=(measurements.shape[0] * measurements.shape[1], 3 * num_poses + 3 * num_landmarks))

        for i in range(num_poses - 1):
            A[i * 3:i * 3 + 3, i * 3:i * 3 + 3] = -np.eye(3)
            A[i * 3:i * 3 + 3, (i + 1) * 3: (i + 1) * 3 + 3] = np.eye(3)

        for i, m in enumerate(associations):
            k = i + num_poses - 1
            A[k * 3: k * 3 + 3, m[0] * 3: m[0] * 3 + 3] = -np.eye(3)
            A[k * 3: k * 3 + 3, (m[1]+num_poses) * 3: (m[1]+num_poses) * 3 + 3] = np.eye(3)

        measurements = measurements.reshape((-1,))
        sA = sparse.csr_matrix(A)
        x = scipy.sparse.linalg.lsqr(A, measurements)

        plot_sparse_matrix(A)

        print(measurements.shape)
        print(x[0])

        return x[0]
