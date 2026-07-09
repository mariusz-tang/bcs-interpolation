"""Utilities for working with Open3D meshes in a torch-focused workflow."""

import pathlib
from functools import cached_property
from typing import Any

import numpy as np
import open3d as o3d
import torch


class TriangleMesh:
    """A convenience wrapper around `open3d.geometry.TriangleMesh`."""

    def __init__(self, mesh: o3d.geometry.TriangleMesh | None = None) -> None:
        """Initialize an instance using an Open3D triangle mesh."""
        if mesh is None:
            mesh = o3d.geometry.TriangleMesh()
        self._mesh = mesh

    @property
    def open3d(self) -> o3d.geometry.TriangleMesh:
        """The underlying open3d mesh."""
        return self._mesh

    @cached_property
    def vertices(self) -> torch.Tensor:
        """Vertex positions.

        Shape: (num_vertices, 3)
        """
        return _tensor(self._mesh.vertices)

    @property
    def num_vertices(self) -> int:
        """Number of vertices."""
        return self.vertices.shape[0]

    @cached_property
    def triangles(self) -> torch.Tensor:
        """Triangle vertex indices.

        Shape: (num_triangles, 3)
        """
        return _tensor(self._mesh.triangles)

    @property
    def num_triangles(self) -> int:
        """Number of triangles."""
        return self.triangles.shape[0]

    @cached_property
    def vertex_normals(self) -> torch.Tensor:
        """Vertex normals, calculated as the average of adjacent face normals.

        Shape: (num_vertices, 3)
        """
        if not self._mesh.has_vertex_normals():
            self._mesh.compute_vertex_normals()
        return _tensor(self._mesh.vertex_normals)

    @cached_property
    def adjacency_list(self) -> list[set[int]]:
        """List of vertex adjacency sets."""
        if not self._mesh.has_adjacency_list():
            self._mesh.compute_adjacency_list()
        return self._mesh.adjacency_list

    def __add__(self, other: Any) -> "TriangleMesh":  # noqa: ANN401 dynamic typing
        """Add `TriangleMesh` instances by joining their geometries together."""
        if type(other) is TriangleMesh:
            return TriangleMesh(self.open3d + other.open3d)
        return NotImplemented

    def _clear(self) -> None:
        self.__dict__.pop("vertices", None)
        self.__dict__.pop("triangles", None)
        self.__dict__.pop("vertex_normals", None)
        self.__dict__.pop("adjacency_list", None)

    def subdivide_midpoint(self, number_of_iterations: int) -> "TriangleMesh":
        """Subdivide the mesh at edge midpoints.

        Returns a new mesh.
        """
        return TriangleMesh(self._mesh.subdivide_midpoint(number_of_iterations))

    def merge_close_vertices(self, eps: float) -> "TriangleMesh":
        """Merge vertices that are within `eps` of each other.

        Acts in-place and returns `self`.
        """
        self._mesh.merge_close_vertices(eps)
        self._clear()
        return self

    def set_vertex_colors(self, colors: torch.Tensor) -> None:
        """Set vertex colors on the underlying open3d mesh."""
        self._mesh.vertex_colors = o3d.utility.Vector3dVector(colors.numpy())


def _tensor(
    a: o3d.utility.Vector3dVector | o3d.utility.Vector3iVector,
) -> torch.Tensor:
    return torch.tensor(np.asarray(a))


def read_from_file(path: pathlib.Path) -> TriangleMesh:
    """Load a mesh from a file."""
    o3d_mesh = o3d.io.read_triangle_mesh(path)
    return TriangleMesh(o3d_mesh)


def write_to_file(
    path: pathlib.Path, mesh: TriangleMesh, write_vertex_colors: bool = False
) -> None:
    """Write a mesh to a file."""
    print(f"Writing mesh to {path}")
    o3d.io.write_triangle_mesh(
        path,
        mesh.open3d,
        write_vertex_normals=False,
        write_vertex_colors=write_vertex_colors,
        write_triangle_uvs=False,
    )


def from_tensors(vertices: torch.Tensor, triangles: torch.Tensor) -> TriangleMesh:
    """Construct a `TriangleMesh` instance from vertex and triangle tensors."""
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices.numpy()),
        o3d.utility.Vector3iVector(triangles.numpy()),
    )
    return TriangleMesh(mesh)


def show(mesh: TriangleMesh) -> None:
    """Visualize a mesh using open3d's visualizer."""
    o3d.visualization.draw_geometries([mesh.open3d])
