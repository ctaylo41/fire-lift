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
        profile = torch.nn.functional.softplus(self.theta)
        return profile.squeeze(0).squeeze(0)

    def world_to_profile_coords(self, points_xyz: Tensor) -> Tensor:
        """Map xyz points to normalized `(r,z)` coordinates for 2D sampling."""
        if points_xyz.shape[-1] != 3:
            raise ValueError(f"Expected final dimension 3, got {points_xyz.shape}")

        r = torch.sqrt(points_xyz[..., 0] ** 2 + points_xyz[..., 1] ** 2)
        z_world = points_xyz[..., 2]

        r_norm = 2.0 * (r / self.max_radius) - 1.0
        z_norm = 2.0 * (z_world - self.z_bounds[0]) / (self.z_bounds[1] - self.z_bounds[0]) - 1.0

        # grid_sample expects the last dimension as (x, y) coordinates.
        return torch.stack([r_norm, z_norm], dim=-1)

    def sample(self, points_xyz: Tensor) -> Tensor:
        """Sample the revolved profile at arbitrary world-space points."""
        coords = self.world_to_profile_coords(points_xyz)
        flat_coords = coords.reshape(-1, 2)

        profile = self.emission_profile()
        profile_grid = profile.unsqueeze(0).unsqueeze(0)

        sample_grid = flat_coords.reshape(1, -1, 1, 2)
        sampled = F.grid_sample(
            profile_grid,
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        values = sampled.squeeze(0).squeeze(-1)
        return values.reshape(*points_xyz.shape[:-1])

    def regularization_field(self) -> Tensor:
        """Apply profile-space priors to the compact physical emission."""
        return self.emission_profile()
