from firelift.volume.dense import VolumeField
import Torch


class FixedDenseVolume(VolumeField):
    def __init__(self,
            resolution: tuple[int, int, int] = (32, 32, 32),
            *,
            init_value: float = -4.0,
            bounds: tuple[float, float] = (-1.0, 1.0),):
        super().__init__()
        self.resolution = resolution
        self.init_value = init_value
        self.bounds = bounds
        
        d, h, w = resolution
        self.theta = torch.full((1,1, d, h, w), init_value)
    
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

    
    
    def regularization_field(self):
        return self.grid