import numpy as np
import open3d as o3d
import time

t1 = time.time()

file = "../kitti_color3d/velodyne_points/data/0000000000.bin"
scan = np.fromfile(file, dtype=np.float32)
scan = scan.reshape((-1, 4))

xyz = scan[:,:3]

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(xyz)
# o3d.visualization.draw_geometries([pcd])

print(scan.shape)

t2 = time.time()
print("Time: ",  t2 - t1)

import os
p = os.listdir('../kitti_color3d/velodyne_points/data/')
l = os.listdir('../kitti_color3d/image_00/data/')
print(len(l), len(p))

