from __future__ import annotations

import torch
from torch import Tensor, nn

from .base import VolumeField


class DenseVolume(VolumeField):
    """Dense learnable scalar emission grid.

    Memory layout:
        theta: [1, 1, D, H, W] == [batch, channel, z, y, x]

    Sampling convention:
        `grid_sample` coordinates are still `(x, y, z)` in [-1, 1].
    """

    def __init__(
        self,
        resolution: tuple[int, int, int] = (32, 32, 32),
        *,
        init_value: float = -4.0,
        bounds: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        self.resolution = resolution
        self.bounds = bounds

        d, h, w = resolution
        self.theta = nn.Parameter(torch.full((1, 1, d, h, w), init_value))

    def emission_grid(self) -> Tensor:
        emission = torch.nn.functional.softplus(self.theta)
        emission = emission.squeeze(0).squeeze(0)
        
        return emission
        

    def world_to_grid(self, points_xyz: Tensor) -> Tensor:
        min_bound, max_bound = self.bounds
        
        range = max_bound - min_bound
        
        normalized = 2 * (points_xyz - min_bound) / range - 1
        
        return normalized
        

    def sample(self, points_xyz: Tensor) -> Tensor:
        normalized_world = self.world_to_grid(points_xyz)
        
        grid = normalized_world.unsqueeze(0)
        sampled = torch.nn.functional.grid_sample(
            self.emission_grid().unsqueeze(0).unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode='zeros',
            align_corners=True
        )
        
        return sampled.squeeze(0).squeeze(0).squeeze(1).squeeze(-1)
     
        

    def regularization_field(self) -> Tensor:
        """Return the physical emission field on which priors act."""
        return self.emission_grid()
