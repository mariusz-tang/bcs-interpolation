"""Blended polynomial surface deformations."""

from collections.abc import Callable

import torch
import torchmin

from bcsi import bps, mesh

from . import bps as deform_bps
from . import polyline
from .typing import Metric


def bps_full(
    start: bps.BlendedPolynomialSurface,
    finish: bps.BlendedPolynomialSurface,
    metric: Metric,
    resolution: int = 0,
    num_frames: int = 4,
    init: bps.BlendedPolynomialSurface | None = None,
    method: str = "newton-cg",
) -> polyline.Polyline[bps.BlendedPolynomialSurface]:
    """Find a two-segment polyline to connect two BPSs using the ARAP metric.

    :param start: The start-point of the polyline.
    :param finish: The end-point of the polyline.
    :param resolution: The resolution to use when evaluating the ARAP metric.
    :param num_frames: The number of frames to use per segment when evaluating
    the ARAP metric.
    :param init: The initial guess for the midpoint of the polyline. The default
    is to linearly interpolate halfway between `start` and `finish`.
    """
    num_vert_coeffs = start.proxy.vertices.numel()

    def make_bps(x: torch.Tensor) -> bps.BlendedPolynomialSurface:
        verts = x[:num_vert_coeffs].reshape(start.proxy.vertices.shape)
        coeffs = x[num_vert_coeffs:].reshape(start.coefficients.shape)
        proxy = mesh.TriangleMesh(verts, start.proxy.triangles)
        return bps.BlendedPolynomialSurface(
            proxy, start.degree, start.global_scale, coeffs, start.beta
        )

    if init is None:
        init = deform_bps.make_frame(start, finish, 0.5)

    x0 = torch.cat([init.proxy.vertices.flatten(), init.coefficients.flatten()])

    return _optimize_intermediate_frame(
        start, finish, make_bps, x0, metric, resolution, num_frames, method=method
    )


def bps_proxy_only(
    start: bps.BlendedPolynomialSurface,
    finish: bps.BlendedPolynomialSurface,
    metric: Metric,
    resolution: int = 0,
    num_frames: int = 4,
    init: bps.BlendedPolynomialSurface | None = None,
    method: str = "newton-cg",
) -> polyline.Polyline[bps.BlendedPolynomialSurface]:
    """Find a two-segment polyline to connect two BPSs using the ARAP metric.

    This function optimizes the proxy only; the coefficients remain unchanged.

    :param start: The start-point of the polyline.
    :param finish: The end-point of the polyline.
    :param resolution: The resolution to use when evaluating the ARAP metric.
    :param num_frames: The number of frames to use per segment when evaluating
    the ARAP metric.
    :param init: The initial guess for the midpoint of the polyline. The default
    is to linearly interpolate halfway between `start` and `finish`.
    """
    if init is None:
        init = deform_bps.make_frame(start, finish, 0.5)

    def make_bps(x: torch.Tensor) -> bps.BlendedPolynomialSurface:
        verts = x.reshape(start.proxy.vertices.shape)
        proxy = mesh.TriangleMesh(verts, start.proxy.triangles)
        return bps.BlendedPolynomialSurface(
            proxy, start.degree, start.global_scale, init.coefficients, start.beta
        )

    x0 = init.proxy.vertices.flatten()

    return _optimize_intermediate_frame(
        start,
        finish,
        make_bps,
        x0,
        metric,
        resolution,
        num_frames,
        xtol=1e-2,
        method=method,
    )


def bps_coefficients_only(
    start: bps.BlendedPolynomialSurface,
    finish: bps.BlendedPolynomialSurface,
    metric: Metric,
    resolution: int = 0,
    num_frames: int = 4,
    init: bps.BlendedPolynomialSurface | None = None,
    method: str = "newton-cg",
) -> polyline.Polyline[bps.BlendedPolynomialSurface]:
    """Find a two-segment polyline to connect two BPSs using the ARAP metric.

    This function optimizes the coefficients only; the proxy remains unchanged.

    :param start: The start-point of the polyline.
    :param finish: The end-point of the polyline.
    :param resolution: The resolution to use when evaluating the ARAP metric.
    :param num_frames: The number of frames to use per segment when evaluating
    the ARAP metric.
    :param init: The initial guess for the midpoint of the polyline. The default
    is to linearly interpolate halfway between `start` and `finish`.
    """
    if init is None:
        init = deform_bps.make_frame(start, finish, 0.5)

    def make_bps(x: torch.Tensor) -> bps.BlendedPolynomialSurface:
        coeffs = x.reshape(start.coefficients.shape)
        return bps.BlendedPolynomialSurface(
            init.proxy, start.degree, start.global_scale, coeffs, start.beta
        )

    x0 = init.coefficients.flatten()

    return _optimize_intermediate_frame(
        start, finish, make_bps, x0, metric, resolution, num_frames, method=method
    )


def _optimize_intermediate_frame(
    start: bps.BlendedPolynomialSurface,
    finish: bps.BlendedPolynomialSurface,
    make_bps_func: Callable[[torch.Tensor], bps.BlendedPolynomialSurface],
    x0: torch.Tensor,
    metric: Metric,
    resolution: int = 0,
    num_frames: int = 4,
    xtol: float = 1e-5,
    method: str = "newton-cg",
) -> polyline.Polyline[bps.BlendedPolynomialSurface]:
    """Find a two-segment polyline to connect two BPSs using the ARAP metric.

    :param start: The start-point of the polyline.
    :param finish: The end-point of the polyline.
    :param make_bps_func: Function that produces a BPS from input data.
    :param x0: The initial guess.
    :param resolution: The resolution to use when evaluating the ARAP metric.
    :param num_frames: The number of frames to use per segment when evaluating
    the ARAP metric.
    :param xtol: average relative error in solution acceptable for convergence.
    """

    def calc_energy(x: torch.Tensor) -> torch.Tensor:
        bps = make_bps_func(x)
        bps.triangle_onering_flips = start.triangle_onering_flips
        bps.triangle_onering_indices = start.triangle_onering_indices
        e = deform_bps.Polyline([start, bps, finish]).symmetric_energy(
            deform_bps.energy_function(metric, resolution),
            2 * num_frames - 1,
        )
        print(e.item())
        return e

    if method in ["bfgs", "l-bfgs"]:
        # Adjust to match the different defaults.
        xtol = xtol * 1e-3

    options = {"xtol": xtol}
    if method == "l-bfgs":
        options["history_size"] = 20

    result = torchmin.minimize(
        calc_energy,
        x0,
        method,
        disp=True,
        options=options,
    )
    intermediate_frame = make_bps_func(result.x)

    return deform_bps.Polyline([start, intermediate_frame, finish])
