"""Package for working with triangular meshes."""

from bcsi.mesh.mesh import TriangleMesh, show

from . import diff, submesh

__all__ = [
    "TriangleMesh",
    "show",
    "diff",
    "submesh",
]
