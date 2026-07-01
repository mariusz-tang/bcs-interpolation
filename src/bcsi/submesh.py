"""Utilities for working with meshes whose vertices are a subset of another's."""

import numpy as np
import open3d as o3d
import torch

from bcsi import bps


def find_vertex_indices(
    submesh: o3d.geometry.TriangleMesh, parent: o3d.geometry.TriangleMesh
) -> torch.Tensor:
    """Calculate indices of vertices found in `submesh` relative to `parent`.

    Returns a (num_vertices,) shape tensor where the `i`th entry represents
    the index of the `i`th vertex in `submesh`, in the vertex list of `parent`.
    """
    subverts = torch.tensor(np.asarray(submesh.vertices))
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


def create_bps_degree_one(
    submesh: o3d.geometry.TriangleMesh, parent: o3d.geometry.TriangleMesh
) -> bps.BlendedPolynomialSurface:
    """Create a BPS using `submesh` as the proxy with information from `parent`.

    The patch functions are unit planes whose normals are determined by the
    normals at each vertex of `submesh` in `parent`.
    """
    vert_indices = find_vertex_indices(submesh, parent)
    base_surface = bps.BlendedPolynomialSurface(submesh, degree=1)

    if not parent.has_vertex_normals():
        parent.compute_vertex_normals()

    parent_normals = torch.tensor(np.asarray(parent.vertex_normals))

    # Map parent vertex normals to child patch function space.
    normals = torch.einsum(
        "vij,vj->vi",
        torch.inverse(base_surface.vertex_rotations).double(),
        parent_normals[vert_indices],
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
    return bps.BlendedPolynomialSurface(submesh, degree=1, coefficients=coefficients)
