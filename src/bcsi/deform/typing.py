"""Type aliases for deformations."""

from collections.abc import Callable

import torch

from bcsi import mesh

type EnergyFunction[T] = Callable[[T, T], torch.Tensor]

type Metric = Callable[[mesh.TriangleMesh, torch.Tensor], torch.Tensor]
