from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base import VolumeField


class AxisymmetricVolume(VolumeField):
    """Emission profile A(z, r) revolved around the world z-axis.

    Instead of optimizing a full `[D,H,W]` volume, optimize a compact
    `[H_z, R]` profile. At a world point `(x,y,z)`, query the profile at
    `r = sqrt(x^2 + y^2)` and height `z`.
    """

    def __init__(
        self,
        profile_resolution: tuple[int, int] = (32, 16),
        *,
        max_radius: float = 1.0,
        z_bounds: tuple[float, float] = (-1.0, 1.0),
        init_value: float = -4.0,
    ) -> None:
        super().__init__()
        h_z, r = profile_resolution
        self.profile_resolution = profile_resolution
        self.max_radius = max_radius
        self.z_bounds = z_bounds
        self.theta = nn.Parameter(torch.full((1, 1, h_z, r), init_value))

    def emission_profile(self) -> Tensor:
        """Return non-negative profile `[H_z,R]`."""
        raise NotImplementedError

    def world_to_profile_coords(self, points_xyz: Tensor) -> Tensor:
        """Map xyz points to normalized `(r,z)` coordinates for 2D sampling.

        TODO:
        - r = sqrt(x^2 + y^2)
        - normalize r from [0, max_radius] to [-1,1]
        - normalize z from z_bounds to [-1,1]
        - remember 2D `grid_sample` expects coordinates `(x,y)`, so using
          `(r,z)` is natural if profile memory is `[z,r]`
        """
        raise NotImplementedError

    def sample(self, points_xyz: Tensor) -> Tensor:
        """Sample the revolved profile at arbitrary world-space points."""
        raise NotImplementedError

    def regularization_field(self) -> Tensor:
        """Apply profile-space priors to the compact physical emission."""
        raise NotImplementedError
