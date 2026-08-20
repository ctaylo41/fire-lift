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


class BentAxisymmetricVolume(AxisymmetricVolume):
    """Axisymmetric profile around an observable x/z centerline.

    The camera looks along world y, so only the image-plane displacement
    ``cx(z)`` is learned. The unobserved depth displacement ``cy(z)`` is fixed
    to zero by the prior.
    """

    def __init__(
        self,
        profile_resolution: tuple[int, int] = (32, 16),
        *,
        max_radius: float = 1.0,
        z_bounds: tuple[float, float] = (-1.0, 1.0),
        init_value: float = -3.0,
        max_center_offset: float = 0.5,
        centerline_points: int = 6,
    ) -> None:
        super().__init__(
            profile_resolution,
            max_radius=max_radius,
            z_bounds=z_bounds,
            init_value=init_value,
        )
        if centerline_points < 2:
            raise ValueError("centerline_points must be at least 2")
        self.max_center_offset = max_center_offset
        self.centerline_points = centerline_points
        self.centerline = nn.Parameter(torch.zeros(1, 1, centerline_points))

    def centerline_at_z(self, z_world: Tensor) -> Tensor:
        """Interpolate the bounded image-plane ``cx(z)`` centerline."""
        z_min, z_max = self.z_bounds
        z_norm = (z_world - z_min) / (z_max - z_min)
        position = z_norm.clamp(0.0, 1.0) * (self.centerline.shape[-1] - 1)
        lower = position.floor().long()
        upper = (lower + 1).clamp(max=self.centerline.shape[-1] - 1)
        fraction = position - lower.to(position.dtype)
        values = self.max_center_offset * torch.tanh(self.centerline)
        lower_values = values[0, 0, lower]
        upper_values = values[0, 0, upper]
        return lower_values * (1.0 - fraction) + upper_values * fraction

    def world_to_profile_coords(self, points_xyz: Tensor) -> Tensor:
        if points_xyz.shape[-1] != 3:
            raise ValueError(f"Expected final dimension 3, got {points_xyz.shape}")
        center_x = self.centerline_at_z(points_xyz[..., 2])
        shifted_x = points_xyz[..., 0] - center_x
        shifted_y = points_xyz[..., 1]
        radius = torch.sqrt(shifted_x.square() + shifted_y.square())
        z_world = points_xyz[..., 2]
        r_norm = 2.0 * (radius / self.max_radius) - 1.0
        z_norm = 2.0 * (z_world - self.z_bounds[0]) / (self.z_bounds[1] - self.z_bounds[0]) - 1.0
        return torch.stack([r_norm, z_norm], dim=-1)

    def regularization_field(self) -> Tensor:
        """Return profile plus the observable x-centerline parameters."""
        return torch.cat([self.emission_profile().reshape(-1), self.centerline.reshape(-1)])


class FourierVolume(VolumeField):
    """Compact first-order angular Fourier volume.

    The learnable profile stores three channels over `(z, r)`. Emission is
    evaluated as `softplus(A0 + A1*cos(theta) + B1*sin(theta))`.
    """

    def __init__(
        self,
        profile_resolution: tuple[int, int] = (32, 16),
        *,
        max_radius: float = 1.0,
        z_bounds: tuple[float, float] = (-1.0, 1.0),
        init_value: float = -3.0,
    ) -> None:
        super().__init__()
        h_z, r = profile_resolution
        self.profile_resolution = profile_resolution
        self.max_radius = max_radius
        self.z_bounds = z_bounds
        self.theta = nn.Parameter(torch.full((1, 3, h_z, r), init_value))

    def world_to_profile_coords(self, points_xyz: Tensor) -> tuple[Tensor, Tensor]:
        if points_xyz.shape[-1] != 3:
            raise ValueError(f"Expected final dimension 3, got {points_xyz.shape}")

        radius = torch.sqrt(points_xyz[..., 0] ** 2 + points_xyz[..., 1] ** 2)
        z_world = points_xyz[..., 2]
        r_norm = 2.0 * (radius / self.max_radius) - 1.0
        z_norm = 2.0 * (z_world - self.z_bounds[0]) / (self.z_bounds[1] - self.z_bounds[0]) - 1.0
        angle = torch.atan2(points_xyz[..., 1], points_xyz[..., 0])
        return torch.stack([r_norm, z_norm], dim=-1), angle

    def emission_profile(self) -> Tensor:
        return torch.nn.functional.softplus(self.theta)

    def sample(self, points_xyz: Tensor) -> Tensor:
        coords, angle = self.world_to_profile_coords(points_xyz)
        sample_grid = coords.reshape(1, -1, 1, 2)
        profiles = self.theta
        sampled = F.grid_sample(
            profiles,
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(0).squeeze(-1)
        sampled = sampled.reshape(3, *points_xyz.shape[:-1])
        logits = (
            sampled[0]
            + sampled[1] * torch.cos(angle)
            + sampled[2] * torch.sin(angle)
        )
        in_bounds = (coords[..., 0].abs() <= 1.0) & (coords[..., 1].abs() <= 1.0)
        return F.softplus(logits) * in_bounds.to(logits.dtype)

    def regularization_field(self) -> Tensor:
        """Return the three compact Fourier coefficient channels."""
        return self.theta.squeeze(0)
