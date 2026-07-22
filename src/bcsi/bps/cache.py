"""Utilities for caching computationally expensive BPS onering data."""

import torch

from bcsi import ROOT_DIR

from . import BlendedPolynomialSurface

CACHE_DIR = ROOT_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def _write(name: str, flips: torch.Tensor, indices: torch.Tensor) -> None:
    torch.save(flips, CACHE_DIR / f"{name}.flips.pt")
    torch.save(indices, CACHE_DIR / f"{name}.indices.pt")


def _read(name: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    flips_path = CACHE_DIR / f"{name}.flips.pt"
    indices_path = CACHE_DIR / f"{name}.indices.pt"

    if not flips_path.exists() or not indices_path.exists:
        return None

    return torch.load(flips_path), torch.load(indices_path)


def onerings(name: str, surface: BlendedPolynomialSurface) -> None:
    """Load onering data from cache, or write it if it doesn't exist."""
    surface: BlendedPolynomialSurface
    if data := _read(name):
        surface.triangle_onering_flips, surface.triangle_onering_indices = data
    else:
        _write(
            name,
            surface.triangle_onering_flips,
            surface.triangle_onering_indices,
        )
