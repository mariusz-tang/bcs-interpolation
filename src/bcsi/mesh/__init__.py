"""Package for working with triangular meshes."""

from bcsi.mesh.mesh import TriangleMesh, from_tensors, show

from . import arap, diff, submesh

__all__ = [
    "TriangleMesh",
    "from_tensors",
    "show",
    "arap",
    "diff",
    "screenshot",
    "submesh",
]
