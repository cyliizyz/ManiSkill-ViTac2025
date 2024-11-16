import pymesh
import open3d as o3d
import numpy as np
import transforms3d as t3d
import meshio
import os


SCALE = 1000.0


def generate_tetrahedral_mesh(gel_name, visualize=False):
    """
    Generates a tetrahedral mesh from an STL file, saves active points, surface points, and faces,
    and optionally visualizes the result.

    Parameters:
        gel_name (str): The name of the gel to process.
        visualize (bool): Whether to visualize the point cloud.

    Returns:
        None
    """
    try:
        # Define file paths
        stl_path = os.path.join("..", "assets", "gel_STL_model", f"{gel_name}.STL")
        output_folder = os.path.join("..", "assets/tac_sensor_meta", gel_name)
        os.makedirs(output_folder, exist_ok=True)

        # Load the mesh using pymesh
        stl_mesh = pymesh.load_mesh(stl_path)

        tetgen = pymesh.tetgen()
        tetgen.points = stl_mesh.vertices / SCALE


        ###########################################################################################
        # You can generate the mesh by adjusting these parameters.
        tetgen.triangles = stl_mesh.faces
        tetgen.split_boundary = True  # Whether to split input boundary, default is True
        tetgen.max_radius_edge_ratio = 1.8  # Default is 2.0
        tetgen.min_dihedral_angle = 0.0  # Default is 0.0
        tetgen.coarsening = False  # Coarsen input tet mesh, default is False
        tetgen.max_tet_volume = 0.00000001  # Default is unbounded
        tetgen.optimization_level = 2  # Ranges from 0 to 10, default is 2
        tetgen.max_num_steiner_points = None  # Default is unbounded
        tetgen.coplanar_tolerance = 1e-8  # Used for coplanar point detection, default is 1e-8
        tetgen.exact_arithmetic = True  # Use exact predicates, default is True
        tetgen.merge_coplanar = True  # Merge coplanar faces and nearby vertices, default is True
        tetgen.weighted_delaunay = False  # Perform weighted Delaunay tetrahedralization, default is False
        tetgen.keep_convex_hull = False  # Keep all tets within convex hull, default is False
        tetgen.verbosity = 1  # Verbosity level from 0 to 4, where 1 is normal output
        ############################################################################################


        # Run tetgen to generate tetrahedral mesh
        tetgen.run()
        tetra_mesh = tetgen.mesh

        nodes = tetra_mesh.nodes.copy()
        faces = tetra_mesh.faces.copy()
        elements = tetra_mesh.elements.copy()

        vn = nodes.shape[0]
        print(f'vn: {vn}, fn: {faces.shape[0]}, en: {elements.shape[0]}')

        # Compute bounding box and center the mesh
        bbox_min = nodes.min(0)
        bbox_max = nodes.max(0)
        center = (bbox_max + bbox_min) / 2
        v = nodes - center[None, :]

        # Apply rotation (currently no rotation applied)
        R = t3d.euler.euler2mat(0, 0., 0.)
        v = (R @ v.T).T

        # Detect active and surface points based on z-axis bounds
        min_z = v.min(0)[2]
        max_z = v.max(0)[2]
        active = v[:, 2] < min_z + 1e-6
        on_surface = v[:, 2] > max_z - 1e-6

        # Save active points and surface points
        active_output = np.zeros(vn, dtype=int)
        active_output[active] = 1
        np.savetxt(os.path.join(output_folder, 'active.txt'), active_output, fmt='%d')

        on_surface_output = np.zeros(vn, dtype=int)
        on_surface_output[on_surface] = 1
        np.savetxt(os.path.join(output_folder, 'on_surface.txt'), on_surface_output, fmt='%d')

        # Save faces to a text file
        np.savetxt(os.path.join(output_folder, 'faces.txt'), faces, fmt='%d')

        # Save tetrahedral mesh in .msh format
        tetra_points = v
        tetra_cells = {"tetra": elements}
        mesh = meshio.Mesh(points=tetra_points, cells=tetra_cells)
        mesh.write(os.path.join(output_folder, 'tet.msh'), binary=False, file_format="gmsh")

        # Optionally visualize the point cloud
        if visualize:
            visualize_mesh(v, vn)

    except Exception as e:
        print(f"An error occurred: {e}")


def visualize_mesh(vertices, vn):
    """
    Visualizes a point cloud using Open3D.

    Parameters:
        vertices (numpy.ndarray): Vertex positions for the point cloud.
        vn (int): Number of vertices.

    Returns:
        None
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)

    colors = np.zeros((vn, 3))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1 / 1000.0)

    o3d.visualization.draw_geometries([pcd, frame])


if __name__ == '__main__':
    generate_tetrahedral_mesh("gelsight_mini", visualize=True)
