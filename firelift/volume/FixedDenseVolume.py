from firelift.volume.dense import VolumeField
import torch
from torch import Tensor


class FixedDenseVolume(VolumeField):
    def __init__(self,
            plume_volume: Tensor,
            resolution: tuple[int, int, int] = (32, 32, 32),
            *,
            init_value: float = -4.0,
            bounds: tuple[float, float] = (-1.0, 1.0),):
        super().__init__()
        self.resolution = resolution
        self.init_value = init_value
        self.bounds = bounds
        d, h, w = resolution
        self.theta = plume_volume
    
    def sample(self, points_xyz):
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

    def world_to_grid(self, points_xyz: Tensor) -> Tensor:
        min_bound, max_bound = self.bounds

        range = max_bound - min_bound

        normalized = 2 * (points_xyz - min_bound) / range - 1
    
        return normalized

    def emission_grid(self) -> Tensor:
        emission = torch.nn.functional.softplus(self.theta)
        emission = emission.squeeze(0).squeeze(0)
        
        return emission

    
    def regularization_field(self):
        return self.emission_grid()