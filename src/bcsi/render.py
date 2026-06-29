"""Utilities for rendering blended chart surfaces as polygonal meshes."""

import math
import random

import numpy as np
import open3d as o3d
import torch

from bcsi import bps


def onering_patch(valence: int, resolution: int) -> o3d.geometry.TriangleMesh:
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

    patch_verts = o3d.core.Tensor(verts)
    patch_tris = o3d.core.Tensor(triangles)

    plane = o3d.t.geometry.TriangleMesh(patch_verts, patch_tris).to_legacy()
    return plane.subdivide_midpoint(resolution)


def triangle_patch(resolution: int) -> o3d.geometry.TriangleMesh:
    """Make a patch corresponding to one triangular face.

    :param resolution: Number of subdivisions to apply. Minimum 0.
    """
    patch_verts = o3d.core.Tensor([[0, 0, 0], [1, 0, 0], [0.5, math.sqrt(3) / 2, 0]])
    patch_tris = o3d.core.Tensor([[0, 1, 2]])

    plane = o3d.t.geometry.TriangleMesh(patch_verts, patch_tris).to_legacy()
    return plane.subdivide_midpoint(resolution)


def blended_polynomial_surface(
    surface: bps.BlendedPolynomialSurface,
    resolution: int,
    color_patches: bool = False,
) -> o3d.geometry.TriangleMesh:
    """Convert BPS to polygonal mesh at a given resolution per face.

    :param resolution: Number of subdivisions to apply to the triangular patch
    representing each face in the proxy mesh.
    :param color_patches: If `True`, assign random colors to each patch.
    """
    # Start with an empty mesh.
    result = o3d.geometry.TriangleMesh()

    # Iteratively add patches corresponding to each face in the proxy.
    patch = triangle_patch(resolution)
    patch_coordinates = torch.tensor(np.asarray(patch.vertices))[:, :2]
    patch_triangles = o3d.core.Tensor(np.asarray(patch.triangles))
    for triangle_id in range(surface.num_triangles):
        mapped_coordinates = o3d.core.Tensor(
            surface.get_blended_patch_vertices(triangle_id, patch_coordinates).numpy()
        )
        mapped_patch = o3d.t.geometry.TriangleMesh(
            mapped_coordinates, patch_triangles
        ).to_legacy()
        if color_patches:
            mapped_patch.paint_uniform_color(_random_color())
        result += mapped_patch

    # Merge the patches into one cohesive mesh.
    return result.merge_close_vertices(eps=1e-6)


def _random_color() -> np.ndarray:
    return np.array([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)])
