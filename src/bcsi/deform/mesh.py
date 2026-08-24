"""Triangle mesh deformations."""

import torch

from bcsi import mesh

from . import polyline
from .typing import EnergyFunction, Metric


class Polyline(polyline.Polyline[mesh.TriangleMesh]):
    """Triangle mesh polyline deformation."""

    @staticmethod
    def _interpolate(
        start: mesh.TriangleMesh, finish: mesh.TriangleMesh, t: float
    ) -> mesh.TriangleMesh:
        dv_dt = finish.vertices - start.vertices
        return mesh.TriangleMesh(start.vertices + t * dv_dt, start.triangles)


def energy_function(
    metric: Metric,
) -> EnergyFunction[mesh.TriangleMesh]:
    """Deformation energy function for a linear deformation between two meshes."""

    def energy(start: mesh.TriangleMesh, finish: mesh.TriangleMesh) -> torch.Tensor:
        dv_dt = (finish.vertices - start.vertices).flatten()
        return metric(start, dv_dt) + metric(finish, dv_dt)

    return energy
