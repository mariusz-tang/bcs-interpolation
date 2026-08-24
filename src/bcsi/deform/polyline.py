"""Generic polyline deformations."""

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

import torch

from .typing import EnergyFunction


class Polyline[T](ABC):
    """A piece-wise linear deformation between keyframes."""

    def __init__(self, frames: Sequence[T]) -> None:
        """Initialize a polyline deformation from a set of keyframes."""
        self.frames = frames

    def get_frame(self, t: float) -> T:
        """Get the frame at time t.

        t is clamped to [0, 1].
        """
        if t >= 1:
            return self.frames[-1]

        # Clamp t from below and scale to the number of segments.
        t = max(0, t) * self.num_segments

        # Find the relevant segment.
        segment_start = math.floor(t)
        segment_progress = t % 1
        return self._interpolate(
            self.frames[segment_start], self.frames[segment_start + 1], segment_progress
        )

    @staticmethod
    @abstractmethod
    def _interpolate(start: T, finish: T, t: float) -> T:
        """Linearly interpolate between `start` and `finish`.

        T should be in [0,1].
        """

    @property
    def num_segments(self) -> int:
        """The number of segments in this polyline."""
        return len(self.frames) - 1

    def _segments(
        self,
    ) -> Iterable[tuple[T, T]]:
        """Return an iterable over the left and right endpoints of each segment."""
        for i in range(self.num_segments):
            yield self.frames[i], self.frames[i + 1]

    def symmetric_energy_distribution(
        self,
        energy_func: EnergyFunction[T],
        num_frames: int = 10,
    ) -> torch.Tensor:
        """Get the energy distribution of this deformation.

        The energy function should take two instances of `T` and return a tensor
        representing the energy of a linear deformation between them.
        """
        result = torch.zeros(num_frames - 1)
        next_frame = self.frames[0]
        for i in range(num_frames - 1):
            frame = next_frame
            next_frame = self.get_frame((i + 1) / (num_frames - 1))
            result[i] = energy_func(frame, next_frame) * (num_frames - 1)

        return result

    def symmetric_energy(
        self, energy_func: EnergyFunction[T], num_frames: int = 10
    ) -> torch.Tensor:
        """Get the energy of this deformation."""
        return self.symmetric_energy_distribution(energy_func, num_frames).sum()

    def subdivide(self) -> "Polyline":
        """Insert a keyframe at the midpoint of each segment.

        Acts in-place and returns `self`.
        """
        new_frames = []
        for lhs, rhs in self._segments():
            new_frames.append(lhs)
            new_frames.append(self._interpolate(lhs, rhs, 0.5))
        new_frames.append(self.frames[-1])

        self.frames = new_frames
        return self
