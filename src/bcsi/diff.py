"""Utilities for comparing meshes."""

import builtins

import numpy as np
import open3d as o3d
import torch

from bcsi import mesh


def vertex_to_vertex(
    source: mesh.TriangleMesh, target: mesh.TriangleMesh
) -> torch.Tensor:
    """Distance from each vertex in `source` to the nearest vertex in `target`.

    Shape: (num_source_vertices)
    """
    points_source = o3d.geometry.PointCloud(source.open3d.vertices)
    points_target = o3d.geometry.PointCloud(target.open3d.vertices)
    distances_o3d = points_source.compute_point_cloud_distance(points_target)
    return torch.tensor(np.asarray(distances_o3d))


def print(diff: torch.Tensor) -> None:
    """Print diff information from a `diff` output tensor."""
    builtins.print(f"  mean: {diff.mean()}")
    builtins.print(f"   std: {diff.std()}")
    builtins.print(f"median: {diff.median()}")
    builtins.print(f"   max: {diff.max()}")
