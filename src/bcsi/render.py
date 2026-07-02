"""Utilities for rendering blended chart surfaces as polygonal meshes."""

import math
import random

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
    color_patches: bool = False,
) -> mesh.TriangleMesh:
    """Convert BPS to polygonal mesh at a given resolution per face.

    :param resolution: Number of subdivisions to apply to the triangular patch
    representing each face in the proxy mesh.
    :param color_patches: If `True`, assign random colors to each patch.
    """
    # Start with an empty mesh.
    result = mesh.TriangleMesh()

    # Iteratively add patches corresponding to each face in the proxy.
    patch = triangle_patch(resolution)
    patch_coordinates = patch.vertices[:, :2]
    for triangle_id in range(surface.proxy.num_triangles):
        mapped_coordinates = surface.get_blended_patch_vertices(
            triangle_id, patch_coordinates
        )
        mapped_patch = mesh.from_tensors(mapped_coordinates, patch.triangles)
        if color_patches:
            mapped_patch.open3d.paint_uniform_color(_random_color())
        result += mapped_patch

    # Merge the patches into one cohesive mesh.
    return result.merge_close_vertices(eps=1e-6)


def _random_color() -> np.ndarray:
    return np.array([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)])
