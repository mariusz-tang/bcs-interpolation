"""Utilities for working with meshes whose vertices are a subset of another's."""

import numpy as np
import open3d as o3d
import torch


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
