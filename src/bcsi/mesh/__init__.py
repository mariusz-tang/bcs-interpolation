"""Package for working with triangular meshes."""

from bcsi.mesh.mesh import TriangleMesh, show

from . import arap, deform, diff, submesh

__all__ = [
    "TriangleMesh",
    "show",
    "arap",
    "deform",
    "diff",
    "screenshot",
    "submesh",
]
