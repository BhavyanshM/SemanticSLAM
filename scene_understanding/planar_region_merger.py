import numpy as np
import os
import matplotlib.pyplot as plt
import math
from TransformUtils import *
from slam.utils import *

class PlanarRegion:
    def __init__(self):
        self.id = -1
        self.center = np.array([0, 0, 0])
        self.normal = np.array([0, 0, 0])
        self.patches = []
        self.segment_indices = [1]
        self.segments = []

    def print(self):
        print("{} {} {} {}".format(self.id, len(self.patches), self.center, len(self.patches)))

    def transform(self, transform):
        center = np.ones((4,))
        center[:3] = self.center
        self.center = (transform @ center)[:3]
        self.normal = transform[:3,:3] @ self.normal
        for i in range(len(self.patches)):
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
            if norms[i] < 0.3:
                self.segment_indices.append(self.segment_indices[i])
            else:
                self.segment_indices.append(self.segment_indices[i] + 1)

        self.segment_indices = np.array(self.segment_indices)

        for i in range(1, self.segment_indices[-1]):
            segment = (np.min(np.argwhere(self.segment_indices == i)), np.max(np.argwhere(self.segment_indices == i)))
            if segment[1] - segment[0] > 1:
                self.segments.append(segment)

        for segment in self.segments:
            print("Segments:", segment, segment[1] - segment[0])

        return norms, self.segments

    def reduce_segment_cosine(self, i):
        points = self.patches[self.segments[i][0]:self.segments[i][1]]
        final_points = []
        for j in range(0, len(points) - 5):
            v1, v2 = points[j+1] - points[j], points[j+2] - points[j+1]
            cosim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            if cosim > 0.9999:
                final_points.append(points[j])
        return final_points

    def reduce_segment_linewise(self, i):

        # TODO: Complete this function for line-based reduction of points.

        points = self.patches[self.segments[i][0]:self.segments[i][1]]
        final_points = []
        for j in range(0, len(points) - 5):
            v1 = points[j+1] - points[j]
            v1 /= np.linalg.norm(v1)
            normal = np.array([-v1[1], v1[0]])
        #     cosim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        #     if cosim > 0.9999:
        #         final_points.append(points[j])
        # return final_points

class PlanarRegionProcessor:
    def __init__(self, path):
        self.path = path
        self.files = sorted(os.listdir(self.path))

        self.fig = plt.figure(figsize=(15, 15))
        plt.xlim(-2.5,2.5)
        plt.ylim(-2.5,2.5)

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
            n_patches = int(strings[1])

            for j in range(n_patches):
                patch = list(map(float, f.readline().split(',')))
                region.patches.append(np.array(patch))

            # region.patches = region.patches[0::3]

            # region.print()

            regions.append(region)
        return regions

    def merge_concave_hulls(self, hull1, hull2):
        final_points = []
        for i in range(len(hull1)):
            point = hull1[i]
            w_n = compute_winding_number(point, hull2)
            print("Winding Number:", w_n)
            if w_n < 0.5:
                final_points.append(point)

        for i in range(len(hull2)):
            point = hull2[i]
            w_n = compute_winding_number(point, hull1)
            if w_n < 0.5:
                final_points.append(point)

        return final_points

    def plot_region(self, regions, i, format='ro', raw=False):
        region = regions[i]
        norms, segments = region.compute_ordered_segments()
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

        # plt.bar(indices, norms)

        # plt.bar(indices, region.segment_indices[:-1])

        # print("Shapes:", points[:,0].shape, colors[:,1].shape)

        # plt.scatter(points[:, 0], points[:, 1], c=colors)

        final = []
        if not(raw):
            for i, segment in enumerate(region.segments):
                edges = region.reduce_segment_cosine(i)
                if len(edges) > 10:
                    edges.append(edges[0])
                    edges = np.array(edges)
                    color = (i*342 % 255 / 255, i*123 % 255 / 255, i*322 % 255 / 255)
                    plt.plot(edges[:,0], edges[:,1], format)

                    final.append(edges)

                    # for u in range(-5, 5):
                    #     for v in range(-5, 5):
                    #         p = np.array([u*0.5, v*0.5])
                    #         plt.plot([p[0]], [p[1]], 'ro')
                    #         w_n = compute_winding_number(p, edges[:, :2])
                    #         print("Winding Number:", p, )
                    #         plt.text(p[0], p[1]+0.04, "{:.2f}".format(w_n), fontsize=15)


        if raw:
            points = np.array([region.patches])[0]
            plt.plot(points[:,0], points[:,1], format, lw=3)

        return final

    def run(self):

        region_set1 = self.load_regions_from_file(2)
        region_set2 = self.load_regions_from_file(4)

        segments1 = self.plot_region(region_set1, 0, 'b-', raw=False)
        # segments2 = self.plot_region(region_set2, 0, 'k-')

        # final = self.merge_concave_hulls(segments1[0], segments2[0])
        #
        # points = np.array([final])[0]
        # plt.plot(points[:, 0], points[:, 1], 'r-')

        plt.show()


if __name__ == "__main__":
    processor = PlanarRegionProcessor('../SceneUnderstanding/Set_06_Circle/')
    processor.run()
