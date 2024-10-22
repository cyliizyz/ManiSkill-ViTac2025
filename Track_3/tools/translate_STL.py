import numpy as np
import trimesh
import open3d as o3d
import transforms3d as t3d
import os


def load_and_transform_mesh(import_path, file_name, rotation_angles=(0, 0, 0), scale=1, translation=(0, 0, 0)):
    """
    Loads an STL file, applies rotation, scaling, and translation transformations.

    Parameters:
        import_path (str): The directory path where the STL file is located.
        file_name (str): The name of the STL file.
        rotation_angles (tuple): Euler angles (in radians) for rotation around x, y, and z axes.
        scale (float): Scaling factor.
        translation (tuple): Translation vector along x, y, and z axes.

    Returns:
        mesh (trimesh.Trimesh): The transformed mesh.
    """
    # Load the mesh
    mesh = trimesh.load(os.path.join(import_path, file_name))

    # Apply rotation using Euler angles
    R = t3d.euler.euler2mat(*rotation_angles)
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = R
    mesh.apply_transform(rotation_matrix)

    # Apply scaling
    scale_matrix = np.eye(4) * scale
    scale_matrix[3, 3] = 1  # Keep homogeneous coordinate unchanged
    mesh.apply_transform(scale_matrix)

    # Apply translation
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = translation
    mesh.apply_transform(translation_matrix)

    return mesh


def export_mesh(mesh, save_path, file_name):
    """
    Exports the transformed mesh to a specified directory.

    Parameters:
        mesh (trimesh.Trimesh): The mesh to export.
        save_path (str): Directory to save the exported file.
        file_name (str): Name of the exported file.
    """
    mesh.export(os.path.join(save_path, file_name))


def visualize_mesh_with_frame(mesh):
    """
    Visualizes the mesh and a coordinate frame using Open3D.

    Parameters:
        mesh (trimesh.Trimesh): The mesh to visualize.
    """
    # Convert the trimesh to Open3D format
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

    # Compute normals for the mesh
    o3d_mesh.compute_vertex_normals()

    # Create a coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])

    # Visualize mesh and coordinate frame
    o3d.visualization.draw_geometries([o3d_mesh, coordinate_frame])


if __name__ == "__main__":
    current_dir = os.getcwd()
    relative_import_path = os.path.join(current_dir, "../assets/gel_STL_model")
    file_name = "gelsight_mini.STL"

    # Transformation parameters
    rotation_angles = (0, 0, 0)  # Rotation angles around x, y, z axes
    scale = 1  # Scaling factor
    translation = (0, 0, 0)  # Translation vector along x, y, z axes

    # Load, transform, and export the mesh
    mesh = load_and_transform_mesh(relative_import_path, file_name, rotation_angles, scale, translation)

    # Export mesh using relative path
    export_mesh(mesh, relative_import_path, file_name)

    # Visualize the mesh
    visualize_mesh_with_frame(mesh)
