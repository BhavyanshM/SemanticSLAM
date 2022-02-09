import open3d as o3d
import numpy as np

class Open3DRenderer:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.rend_opt = self.vis.get_render_option()
        self.rend_opt.mesh_show_back_face = True
        self.rend_opt.background_color = np.asarray([0, 0, 0])
        self.pcd = o3d.geometry.PointCloud()
        self.axes = [o3d.geometry.TriangleMesh.create_coordinate_frame(origin=(0, 0, 0))]
        self.line_set = o3d.geometry.LineSet()

        self.vis.add_geometry(self.line_set)
        self.vis.add_geometry(self.axes[0])
        self.vis.add_geometry(self.pcd)

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

    def submit_quad(self, points):
        mesh = o3d.geometry.TriangleMesh()
        np_vertices = np.array([[2, 2, 0],
                                [5, 2, 0],
                                [5, 5, 0],
                                [2, 5, 0]])
        np_triangles = np.array([[0, 1, 2], [0,2,3]]).astype(np.int32)

        mesh.vertices = o3d.utility.Vector3dVector(np_vertices)
        mesh.triangles = o3d.utility.Vector3iVector(np_triangles)
        self.insert_geometry(mesh)

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

