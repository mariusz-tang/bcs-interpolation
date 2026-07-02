"""Utilities for working with meshes whose vertices are a subset of another's.

We refer to the submeshes as 'children' and the original meshes as 'parents'.
"""

import numpy as np
import open3d as o3d
import torch

from bcsi import bps


def find_vertex_correspondences(
    child: o3d.geometry.TriangleMesh, parent: o3d.geometry.TriangleMesh
) -> torch.Tensor:
    """Calculate indices of vertices found in `child` relative to `parent`.

    Returns a (num_vertices,) shape tensor where the `i`th entry represents
    the index of the `i`th vertex in `child`, in the vertex list of `parent`.
    """
    subverts = torch.tensor(np.asarray(child.vertices))
    parent_verts = torch.tensor(np.asarray(parent.vertices))

    num_parent_verts = parent_verts.shape[0]

    # Repeat the sub-vertices so we can compare them against ALL vertices in
    # the parent by broadcasting.
    matches = (
        subverts.repeat(num_parent_verts, 1, 1).transpose(0, 1).isclose(parent_verts)
    )
    matches_complete_vertex = matches.all(dim=2)

    result = matches_complete_vertex.argwhere()[:, 1]

    if result.shape[0] != subverts.shape[0]:
        raise ValueError(
            f"unable to find all vertices from {subverts} in {parent_verts}"
        )

    return result


def new_frame(
    child: o3d.geometry.TriangleMesh,
    correspondences: torch.Tensor,
    parent: o3d.geometry.TriangleMesh,
) -> o3d.geometry.TriangleMesh:
    """Return a new mesh obtained by posing `child` according to `parent`.

    :param correspondences: Vertex correspondences between `child` and `parent`,
    as defined in `find_vertex_correspondences`.
    """
    parent_vertices = torch.tensor(np.asarray(parent.vertices))
    new_vertices = parent_vertices[correspondences]
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(new_vertices.numpy()), child.triangles
    )


def create_bps_degree_one(
    child: o3d.geometry.TriangleMesh,
    parent: o3d.geometry.TriangleMesh,
    correspondences: torch.Tensor,
) -> bps.BlendedPolynomialSurface:
    """Create a BPS using `child` as the proxy with information from `parent`.

    The patch functions are unit planes whose normals are determined by the
    normals at each vertex of `child` in `parent`.

    :param correspondences: Vertex correspondences between `child` and `parent`,
    as defined in `find_vertex_correspondences`.
    """
    base_surface = bps.BlendedPolynomialSurface(child, degree=1)

    if not parent.has_vertex_normals():
        parent.compute_vertex_normals()

    parent_normals = torch.tensor(np.asarray(parent.vertex_normals))

    # Map parent vertex normals to child patch function space.
    normals = torch.einsum(
        # Multiply by the transpose instead of explicitly calculating inverse rotations.
        "vji,vj->vi",
        base_surface.vertex_rotations.double(),
        parent_normals[correspondences],
    )

    # Project x-direction onto normal plane.
    e_1 = torch.tensor([1, 0, 0])
    e_1_projected = e_1 - normals[:, 0:1] * normals
    e_1_normalized = e_1_projected / torch.linalg.vector_norm(
        e_1_projected, dim=1, keepdim=True
    )

    # Calculate y-direction by cross product.
    e_2 = torch.linalg.cross(normals, e_1_normalized)

    # Create coefficients matrix.
    coefficients = torch.zeros_like(base_surface.coefficients)
    coefficients[..., 1] = e_1_normalized
    coefficients[..., 2] = e_2

    # Create BPS with the new coefficients.
    return bps.BlendedPolynomialSurface(child, degree=1, coefficients=coefficients)
