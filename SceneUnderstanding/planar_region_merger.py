import numpy as np
import os
import matplotlib.pyplot as plt
import math
from TransformUtils import *

class PlanarRegion:
    def __init__(self):
        self.id = -1
        self.center = np.array([0, 0, 0])
        self.normal = np.array([0, 0, 0])
        self.n_patches = 0
        self.patches = []
        self.segment_indices = [1]

    def print(self):
        print("{} {} {} {}".format(self.id, self.n_patches, self.center, len(self.patches)))

    def transform(self, transform):
        center = np.ones((4,))
        center[:3] = self.center
        self.center = (transform @ center)[:3]
        self.normal = transform[:3,:3] @ self.normal
        for i in range(self.n_patches):
            patch = np.ones((4,))
            patch[:3] = self.patches[i]
            self.patches[i] = (transform @ patch)[:3] - self.center

    def transform_normal_z_up(self):
        angle = math.acos(math.fabs(np.dot(self.normal, np.array([0, 0, 1]))) / np.linalg.norm(self.normal))
        transform = get_rotation_x(-angle)
        self.transform(transform)

    def compute_ordered_segments(self):
        patches = np.array(self.patches)

        norms = np.linalg.norm(patches[:-1,:] - patches[1:,:], axis=1)

        for i in range(patches.shape[0] - 2):
            if norms[i] < 0.2:
                self.segment_indices.append(self.segment_indices[i])
            else:
                self.segment_indices.append(self.segment_indices[i] + 1)

        self.segment_indices = np.array(self.segment_indices)

        print("Segments:", self.segment_indices.shape)


class PlanarRegionProcessor:
    def __init__(self, path):
        self.path = path
        self.files = sorted(os.listdir(self.path))

        self.fig = plt.figure(figsize=(18, 18))
        # plt.xlim(-3,3)
        # plt.ylim(-3,3)

    def load_regions_from_file(self, i):
        regions = []
        f = open(self.path + self.files[i])
        print("\nLoading Regions From File: ", self.files[i])
        self.num_regions = int(f.readline().replace('\n', '').split(':')[1])

        for i in range(self.num_regions):

            region = PlanarRegion()

            strings = f.readline().replace('\n', '').split(':')
            region.id = int(strings[1])

            strings = f.readline().replace('\n', '').split(':')
            region.center = np.array(list(map(float, strings[1].split(','))))

            strings = f.readline().replace('\n', '').split(':')
            region.normal = np.array(list(map(float, strings[1].split(','))))

            strings = f.readline().replace('\n', '').split(':')
            region.n_patches = int(strings[1])

            for j in range(region.n_patches):
                patch = list(map(float, f.readline().split(',')))
                region.patches.append(np.array(patch))

            region.patches = region.patches[0::2]
            region.n_patches = len(region.patches)

            # region.print()

            regions.append(region)
        return regions

    def plot_region(self, regions, i, format='ro'):
        region = regions[i]
        region.compute_ordered_segments()
        region.transform_normal_z_up()
        points = np.array([region.patches])[0]
        points = points[:, :2]
        print(points.shape)


        patches = np.array(region.patches)
        indices = np.linspace(1, patches.shape[0] - 1, patches.shape[0] - 1)
        region.segment_indices = np.append(region.segment_indices, 1)
        colors = np.zeros(shape=(patches.shape[0], 3))

        colors[:,0] = region.segment_indices * 342 % 255 / 255
        colors[:, 1] = region.segment_indices * 123 % 255 / 255
        colors[:, 2] = region.segment_indices * 225 % 255 / 255

        # plt.bar(indices, region.segment_indices)

        print("Shapes:", points[:,0].shape, colors[:,1].shape)

        plt.scatter(points[:, 0], points[:, 1], c=colors)

    def run(self):

        region_set1 = self.load_regions_from_file(0)
        region_set2 = self.load_regions_from_file(5)

        self.plot_region(region_set1, 0, 'ro')
        # self.plot_region(region_set2, 0, 'bo')
        plt.show()


if __name__ == "__main__":
    processor = PlanarRegionProcessor('../SceneUnderstanding/Set_06_Circle/')
    processor.run()
