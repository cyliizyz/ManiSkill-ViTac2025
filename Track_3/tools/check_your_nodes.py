import meshio
import open3d as o3d
import numpy as np
import os


def load_mesh_and_labels(mesh_path, labels_path):
    """
    Loads the mesh and surface labels from the given file paths.

    Parameters:
        mesh_path (str): Path to the mesh file.
        labels_path (str): Path to the labels file.

    """
    mesh = meshio.read(mesh_path)
    labels = np.loadtxt(labels_path)
    return mesh.points, labels


def create_point_cloud(points, labels):
    """
    Creates a point cloud object with color based on labels.

    Parameters:
        points (numpy.ndarray): Points from the mesh.
        labels (numpy.ndarray): Surface labels.

    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    colors = np.zeros((points.shape[0], 3))
    colors[labels == 1] = [1, 0, 0]  # Red for label 1
    colors[labels != 1] = [0, 0, 1]  # Blue for other labels
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def custom_draw_geometry_with_options(pcd, frame):
    """
    Custom visualization of point cloud and coordinate frame with specific render options.

    Parameters:
        pcd (o3d.geometry.PointCloud): Point cloud to visualize.
        frame (o3d.geometry.TriangleMesh): Coordinate frame to visualize.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.add_geometry(pcd)
    vis.add_geometry(frame)

    render_option = vis.get_render_option()
    render_option.point_size = 5.0
    render_option.background_color = np.array([1, 1, 1])

    ctr = vis.get_view_control()
    ctr.set_front([0.0, 0.0, -1.0])
    ctr.set_lookat([0.0, 0.0, 0.0])
    ctr.set_up([0.0, 1.0, 0.0])
    ctr.set_zoom(0.8)

    render_option.light_on = True

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    # Define paths to mesh and labels files
    mesh_path = os.path.join("..", "assets", "gelsight_mini_e430", "tet.msh")
    labels_path = os.path.join("..", "assets", "gelsight_mini_e430", "on_surface.txt")

    # Load mesh points and labels
    points, labels = load_mesh_and_labels(mesh_path, labels_path)

    # Create point cloud and coordinate frame
    pcd = create_point_cloud(points, labels)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1 / 1000.0)

    # Visualize point cloud and frame
    custom_draw_geometry_with_options(pcd, frame)
