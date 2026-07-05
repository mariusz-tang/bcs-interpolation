"""Utilities for rendering blended chart surfaces as polygonal meshes."""

import math

import numpy as np
import torch

from bcsi import bps, mesh


def onering_patch(valence: int, resolution: int) -> mesh.TriangleMesh:
    """Make a patch representing an entire one-ring of specified valence.

    :param resolution: Number of subdivisions to apply. Minimum 0.
    """
    verts: list[list[float]] = [[0, 0, 0]]
    triangles = []

    for i in range(valence):
        verts.append(
            [math.cos(np.pi * 2 * i / valence), math.sin(np.pi * 2 * i / valence), 0]
        )
        triangles.append([0, i + 1, (i + 1) % valence + 1])

    plane = mesh.from_tensors(torch.tensor(verts), torch.tensor(triangles))
    return plane.subdivide_midpoint(resolution)


def triangle_patch(resolution: int) -> mesh.TriangleMesh:
    """Make a patch corresponding to one triangular face.

    :param resolution: Number of subdivisions to apply. Minimum 0.
    """
    patch_verts = torch.tensor([[0, 0, 0], [1, 0, 0], [0.5, math.sqrt(3) / 2, 0]])
    patch_tris = torch.tensor([[0, 1, 2]])

    plane = mesh.from_tensors(patch_verts, patch_tris)
    return plane.subdivide_midpoint(resolution)


def blended_polynomial_surface(
    surface: bps.BlendedPolynomialSurface,
    resolution: int,
) -> mesh.TriangleMesh:
    """Convert BPS to polygonal mesh at a given resolution per face.

    :param resolution: Number of subdivisions to apply to the triangular patch
    representing each face in the proxy mesh.
    """
    patch = triangle_patch(resolution)
    # Ignore the z coordinate, which is zero everywhere.
    patch_coordinates = patch.vertices[:, :2]
    # Calculate all vertex positions and flatten the result.
    vertices = surface.get_blended_patch_vertices(patch_coordinates).reshape(-1, 3)
    # Duplicate the topology tensor for each patch, increasing the vertex indices
    # by the number of vertices per patch each time. Finally, flatten the result.
    triangles = (
        patch.triangles.tile(surface.proxy.num_triangles, 1, 1)
        + torch.ones(surface.proxy.num_triangles, patch.num_triangles, 3)
        * torch.arange(surface.proxy.num_triangles)[:, None, None]
        * patch.num_vertices
    ).reshape(-1, 3)
    result = mesh.from_tensors(vertices, triangles)

    # Merge the patches into one cohesive mesh.
    return result.merge_close_vertices(eps=1e-6)
