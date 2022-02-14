import open3d as o3d
import numpy as np
from utils import *
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import copy

class Open3DRenderer:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.rend_opt = self.vis.get_render_option()
        self.rend_opt.mesh_show_back_face = True
        self.rend_opt.background_color = np.asarray([0, 0, 0])
        self.rend_opt.light_on = True
        self.pcd = o3d.geometry.PointCloud()
        self.axes = [o3d.geometry.TriangleMesh.create_coordinate_frame(origin=(0, 0, 0))]
        self.line_set = o3d.geometry.LineSet()

        self.vis.add_geometry(self.line_set)
        self.vis.add_geometry(self.axes[0])
        self.vis.add_geometry(self.pcd)

        # gui.Application.instance.initialize()
        # self.window = gui.Application.instance.create_window("use_point_light", 500, 500)
        # self.scene_widget = gui.SceneWidget()
        # self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        # self.window.add_child(self.scene_widget)
        # self.scene_widget.scene.scene.add_point_light('light', [1, 1, 1], -3 * np.array([-1,-1,0]), 1e8, 1e2, True)

        self.control = self.vis.get_view_control()

        self.control.unset_constant_z_near()

    def submit_points(self, xyz, colors=None):
        if colors is not None:
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
        self.pcd.points = o3d.utility.Vector3dVector(xyz)

    def submit_lines(self, points, lines, line_colors):
        self.line_set.points = o3d.utility.Vector3dVector(points)
        self.line_set.lines = o3d.utility.Vector2iVector(lines)
        self.line_set.colors = o3d.utility.Vector3dVector(line_colors)

    def submit_pose(self, pose):
        t = pose[:,3]
        self.axes.append(o3d.geometry.TriangleMesh.create_coordinate_frame(origin=(t[0], t[1], t[2])))
        self.insert_geometry(self.axes[-1])

    def submit_sphere(self, center, radius=0.1):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius, 20)
        # sphere.translate(1,1,1)
        mesh = copy.deepcopy(sphere).translate((center[0], center[1], center[2]))
        self.insert_geometry(mesh)


    def submit_quad(self, point, normal, scale_x=0.5, scale_y=0.5, color=[0.5, 0.1, 0.6]):
        pi = get_plane(point, normal)
        xy1 = (point[0] + scale_x, point[1] + scale_y)
        xy2 = (point[0] + scale_x, point[1] - scale_y)
        xy3 = (point[0] - scale_x, point[1] - scale_y)
        xy4 = (point[0] - scale_x, point[1] + scale_y)
        z1, z2, z3, z4 = get_plane_z(xy1, pi), get_plane_z(xy2, pi), get_plane_z(xy3, pi), get_plane_z(xy4, pi)
        mesh = o3d.geometry.TriangleMesh()
        np_vertices = np.array([[xy1[0], xy1[1], z1],
                                [xy2[0], xy2[1], z2],
                                [xy3[0], xy3[1], z3],
                                [xy4[0], xy4[1], z4]])
        np_triangles = np.array([[0, 1, 2], [0,2,3]]).astype(np.int32)
        mesh.vertices = o3d.utility.Vector3dVector(np_vertices)
        mesh.triangles = o3d.utility.Vector3iVector(np_triangles)
        mesh.paint_uniform_color([color[0], color[1], color[2]])
        self.insert_geometry(mesh)

    def submit_polytope(self, points):
        mesh = o3d.geometry.TriangleMesh()

        np_vertices = []
        for i in range(points.shape[0]):
            np_vertices.append(points[i].tolist())

        np_triangles = np.array([[0, 1, 2],
                                 [0, 2, 3],
                                 [0, 4, 5],
                                 [0, 5, 1],
                                 [1, 5, 6],
                                 [1, 6, 2],
                                 [2, 6, 7],
                                 [2, 7, 3],
                                 [3, 7, 4],
                                 [3, 4, 0],
                                 [4, 5, 6],
                                 [4, 6, 7]
                                 ]).astype(np.int32)


        mesh.vertices = o3d.utility.Vector3dVector(np_vertices)
        mesh.triangles = o3d.utility.Vector3iVector(np_triangles)
        mesh.paint_uniform_color([0.5, 0.5, 0.5])
        self.insert_geometry(mesh)

        for i in range(points.shape[0]):
            self.submit_sphere(points[i])

    def insert_geometry(self, mesh):
        params = self.control.convert_to_pinhole_camera_parameters()
        self.vis.add_geometry(mesh)
        self.control.convert_from_pinhole_camera_parameters(params)

    def update(self):
        self.vis.update_geometry(self.line_set)
        self.vis.update_geometry(self.pcd)
        for pose in self.axes:
            self.vis.update_geometry(pose)
        self.vis.update_renderer()
        if not self.vis.poll_events():
            exit()

