from __future__ import annotations

from abc import ABC, abstractmethod
import torch
from torch import Tensor, nn


class VolumeField(nn.Module, ABC):
    """Abstract emission field queried at arbitrary world-space points."""

    @abstractmethod
    def sample(self, points_xyz: Tensor) -> Tensor:
        """Return non-negative emission at world-space points.

        Args:
            points_xyz: [..., 3] in `(x, y, z)` order.

        Returns:
            emission: [...] scalar emission values.
        """
        raise NotImplementedError
        
        

    @abstractmethod
    def regularization_field(self) -> Tensor:
        """Return the natural parameter-space field on which TV/sparsity act.

        DenseVolume -> dense `[D,H,W]` emission grid.
        AxisymmetricVolume -> `[H,R]` revolved profile.
        """
        raise NotImplementedError

    def materialize(
        self,
        resolution: tuple[int, int, int] = (32, 32, 32),
        bounds: tuple[float, float] = (-1.0, 1.0),
    ) -> Tensor:

        x_range = torch.linspace(bounds[0],bounds[1], resolution[2])
        y_range = torch.linspace(bounds[0],bounds[1], resolution[1])
        z_range = torch.linspace(bounds[0], bounds[1], resolution[0])
        
        d, y_grid, x_grid = torch.meshgrid(z_range, y_range, x_range, indexing="ij")
        
        
        combined = torch.stack([x_grid, y_grid, d], dim=-1)
        
        sampled = self.sample(combined)

        return sampled
