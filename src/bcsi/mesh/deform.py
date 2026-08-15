"""Shape space deformation."""

import math

import torch

from . import TriangleMesh, arap


def make_frame(start: TriangleMesh, finish: TriangleMesh, t: float) -> TriangleMesh:
    """Construct an intermediate BPS by linear interpolation."""
    dv_dt = finish.vertices - start.vertices
    return TriangleMesh(start.vertices + t * dv_dt, start.triangles)


def energy(
    start: TriangleMesh,
    finish: TriangleMesh,
    lamda: float = 1e-6,
) -> torch.Tensor:
    """Calculate deformation energy (for a linear deformation) between two frames."""
    dv_dt = (finish.vertices - start.vertices).flatten()
    return arap.metric(start, dv_dt, lamda) + arap.metric(finish, dv_dt, lamda)


class Polyline:
    """A piece-wise linear deformation between keyframes."""

    def __init__(self, *frames: TriangleMesh) -> None:
        """Initialize a polyline deformation from a set of keyframes."""
        self.frames = frames

    def get_frame(self, t: float) -> TriangleMesh:
        """Get the frame at time t.

        t is clamped to [0, 1].
        """
        if t >= 1:
            return self.frames[-1]

        # Clamp t from below and scale to the number of segments.
        t = max(0, t) * self.num_segments

        # Find the relevant linear segment.
        segment_start = math.floor(t)
        segment_progress = t % 1
        return make_frame(
            self.frames[segment_start], self.frames[segment_start + 1], segment_progress
        )

    @property
    def num_segments(self) -> int:
        """The number of segments in this polyline."""
        return len(self.frames) - 1

    def energy_distribution(
        self, num_frames: int = 2, lamda: float = 1e-6
    ) -> torch.Tensor:
        """Get the energy distribution of this deformation.

        :param num_frames: Number of frames at which to evaluate the energy.
        """
        result = torch.zeros(num_frames - 1)
        next_frame = self.frames[0]
        for i in range(num_frames - 1):
            frame = next_frame
            next_frame = self.get_frame((i + 1) / (num_frames - 1))
            result[i] = energy(frame, next_frame, lamda=lamda) * (num_frames - 1)

        return result

    def energy(self, num_frames: int = 2, lamda: float = 1e-6) -> torch.Tensor:
        """Get the energy of this deformation.

        :param num_frames: Number of frames at which to evaluate the energy.
        """
        return self.energy_distribution(num_frames, lamda).sum()
