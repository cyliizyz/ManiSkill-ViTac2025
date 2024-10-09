from typing import List, Tuple

import os

import sapien
import torch
import numpy as np
from sapienipc.ipc_component import IPCABDComponent
from sapienipc.ipc_utils.ipc_mesh import IPCTetMesh, IPCTriMesh


def build_sapien_entity_ABD_Tri(msh_path: str,
                                torch_device: str,
                                density: float = 1000.0,
                                friction: float = 0.5,
                                color: List[float] = [0.7, 0.7, 0.7, 1.0],
                                no_render: bool = False) -> Tuple[
    sapien.Entity, IPCABDComponent]:
    abd_component = IPCABDComponent()
    abd_component.set_tri_mesh(IPCTriMesh(filename=msh_path, scale=0.001))
    abd_component.set_density(density)
    abd_component.set_friction(friction)

    if not no_render:
        render_component = sapien.render.RenderCudaMeshComponent(
            abd_component.tri_mesh.n_vertices, abd_component.tri_mesh.n_surface_triangles
        )
        render_component.set_vertex_count(abd_component.tri_mesh.n_vertices)
        render_component.set_triangle_count(abd_component.tri_mesh.n_surface_triangles)
        render_component.set_triangles(abd_component.tri_mesh.surface_triangles)
        render_component.set_material(sapien.render.RenderMaterial(
            base_color=color,
            specular=0.8,
            roughness=0.5,
            metallic=0.1,
        ))
    # render_component.set_data_source(abd_component)

    entity = sapien.Entity()
    if not no_render:
        entity.add_component(render_component)
    entity.add_component(abd_component)

    return entity, abd_component


def build_sapien_entity_ABD_Tet(msh_path: str,
                                torch_device: str,
                                density: float = 1000.0,
                                friction: float = 0.5,
                                color: List[float] = [0.7, 0.7, 0.7, 1.0],
                                no_render: bool = False) -> Tuple[
    sapien.Entity, IPCABDComponent]:
    abd_component = IPCABDComponent()
    abd_component.set_tet_mesh(IPCTetMesh(filename=msh_path))
    abd_component.set_density(density)
    abd_component.set_friction(friction)

    if not no_render:
        render_component = sapien.render.RenderCudaMeshComponent(
            abd_component.tet_mesh.n_vertices, abd_component.tet_mesh.n_surface_triangles
        )
        render_component.set_vertex_count(abd_component.tet_mesh.n_vertices)
        render_component.set_triangle_count(abd_component.tet_mesh.n_surface_triangles)
        render_component.set_triangles(abd_component.tet_mesh.surface_triangles)
        render_component.set_material(sapien.render.RenderMaterial(
            base_color=color,
            specular=0.8,
            roughness=0.5,
            metallic=0.1,
        ))
    # render_component.set_data_source(abd_component)

    entity = sapien.Entity()
    if not no_render:
        entity.add_component(render_component)
    entity.add_component(abd_component)

    return entity, abd_component


def build_sapien_entity_ABD(msh_path: str,
                            torch_device: str,
                            density: float = 1000.0,
                            friction: float = 0.5,
                            color: List[float] = [0.7, 0.7, 0.7, 1.0],
                            no_render: bool = False) -> Tuple[
    sapien.Entity, IPCABDComponent]:
    ext = os.path.splitext(msh_path)[-1]
    if ext == ".msh":
        return build_sapien_entity_ABD_Tet(msh_path, torch_device, density, friction, color, no_render)
    elif ext == '.STL':
        return build_sapien_entity_ABD_Tri(msh_path, torch_device, density, friction, color, no_render)
    else:
        raise TypeError(f"Unsupported file extension {ext}: {msh_path}")


def cv2ex2pose(ex):
    cv2sapien = np.array([[0., 0., 1., 0.],
                          [-1., 0., 0., 0.],
                          [0., -1., 0., 0.],
                          [0., 0., 0., 1.]], dtype=np.float32)

    pose = ex @ np.linalg.inv(cv2sapien)

    return sapien.Pose(pose)
