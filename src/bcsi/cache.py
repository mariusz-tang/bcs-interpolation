"""Utilities for caching computationally expensive BPS onering data."""

import pathlib

import torch

from bcsi import bps, cli


def _cache_dir() -> pathlib.Path:
    cache_dir = cli.get_output_dir("cache")
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def _write(name: str, flips: torch.Tensor, indices: torch.Tensor) -> None:
    torch.save(flips, _cache_dir() / f"{name}.flips.pt")
    torch.save(indices, _cache_dir() / f"{name}.indices.pt")


def _read(name: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    flips_path = _cache_dir() / f"{name}.flips.pt"
    indices_path = _cache_dir() / f"{name}.indices.pt"

    if not flips_path.exists() or not indices_path.exists:
        return None

    return torch.load(flips_path), torch.load(indices_path)


def bps_onerings(name: str, surface: bps.BlendedPolynomialSurface) -> None:
    """Load onering data from cache, or write it if it doesn't exist."""
    from bcsi import bps

    surface: bps.BlendedPolynomialSurface
    if data := _read(name):
        surface.triangle_onering_flips, surface.triangle_onering_indices = data
    else:
        _write(
            name,
            surface.triangle_onering_flips,
            surface.triangle_onering_indices,
        )
