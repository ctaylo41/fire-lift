from __future__ import annotations

import torch
from torch import Tensor

from .camera import OrthographicCamera, RayBundle
from firelift.volume.base import VolumeField


def sample_along_rays(
    rays: RayBundle,
    *,
    near: float,
    far: float,
    n_samples: int,
) -> tuple[Tensor, Tensor]:
    dt = (far - near) / n_samples
    t = near + (torch.arange(n_samples, dtype=rays.origins.dtype, device=rays.origins.device) + 0.5) * dt
    origins = rays.origins
    dirs = rays.directions
    
    points = origins[..., None, :] + t.view(*([1] * (origins.ndim - 1)), -1, 1) * dirs[..., None, :]
    return points, t


def emission_integral(samples: Tensor, t: Tensor) -> Tensor:
    dt = (t[-1] - t[0]) / (t.numel() - 1)
    image = samples.sum(dim=-1) * dt
    return image

def render_emission(
    volume: VolumeField,
    camera: OrthographicCamera,
    height: int,
    width: int,
    *,
    n_samples: int = 96,
    ray_chunk: int | None = None,
) -> Tensor:
    """Render an emission-only grayscale image from a volume.

    TODO:
        1. Generate rays.
        2. Sample points between camera.near and camera.far.
        3. Query `volume.sample(points_xyz)`.
        4. Integrate emission along the sample dimension.
        5. Optionally add ray chunking only if memory actually becomes an issue.

    This function should contain no fitting/optimizer logic.
    """
    
    rays = camera.generate_rays(height, width)
    sampled_points, t = sample_along_rays(rays, near=camera.near, far=camera.far, n_samples=n_samples)
    emission_samples = volume.sample(sampled_points)
    image = emission_integral(emission_samples, t)
    return image
