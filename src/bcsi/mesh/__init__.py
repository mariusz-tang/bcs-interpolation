"""Package for working with triangular meshes."""

from bcsi.mesh.mesh import TriangleMesh, show

from . import arap, diff, submesh

__all__ = [
    "TriangleMesh",
    "show",
    "arap",
    "diff",
    "screenshot",
    "submesh",
]
