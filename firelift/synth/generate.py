from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor
import math

@dataclass
class GaussianBlob:
    """Parameters of one anisotropic synthetic emission blob."""

    center_xyz: Tensor       # [3]
    scales_xyz: Tensor       # [3], positive
    amplitude: float


def make_world_grid(
    resolution: tuple[int, int, int],
    *,
    bounds: tuple[float, float] = (-1.0, 1.0),
    device: torch.device | str = "cpu",
) -> Tensor:
    """Return xyz coordinates with shape `[D,H,W,3]`.

    TODO:
        Tensor memory is `[z,y,x]` but the final coordinate vector must be
        `(x,y,z)`.
    """
    x_range = torch.linspace(bounds[0],bounds[1], resolution[2], device=device)
    y_range = torch.linspace(bounds[0],bounds[1], resolution[1], device=device)
    z_range = torch.linspace(bounds[0], bounds[1], resolution[0], device=device)
    
    d, y_grid, x_grid = torch.meshgrid(z_range, y_range, x_range, indexing="ij")
    
    
    combined = torch.stack([x_grid, y_grid, d], dim=-1)
    
    return combined


def gaussian_blob_field(points_xyz: Tensor, blob: GaussianBlob) -> Tensor:
    """Evaluate one axis-aligned anisotropic 3D Gaussian emission blob."""
    offset = points_xyz - blob.center_xyz
    normalized = offset / blob.scales_xyz
    exponent = -0.5 * (normalized ** 2).sum(dim=-1)
    return blob.amplitude * torch.exp(exponent)

def sample_blob_params(
        radial_spread,
        z_range,
        scale_xy_range,
        scale_z_range,
        taper,
        num_samples: int, 
        axisymmetric,
        axis_jitter,
        anisotropy,
        device, 
        dtype
    ):
    def uniform_range(low, high, size):
        return (high-low) * torch.rand(size, device=device, dtype=dtype) + low
    
    z = uniform_range(z_range[0], z_range[1],(num_samples,))
    

    z_min, z_max = z_range
    normalized_height = (z - z_min) / (z_max - z_min)
    taper_factor = 1.0 - taper * normalized_height

    if axisymmetric:
        x = torch.zeros_like(z)
        y = torch.zeros_like(z)
        if axis_jitter > 0.0:
            x = torch.randn_like(z) * axis_jitter
            y = torch.randn_like(z) * axis_jitter
    else:
            
        max_r = radial_spread * taper_factor
        
        angle = uniform_range(0, 2* math.pi, (num_samples,))
        
        u = torch.rand((num_samples,), device=device, dtype=dtype)
        radius = max_r * torch.sqrt(u)
        x = radius * torch.cos(angle)
        y = radius * torch.sin(angle)
        
    
    
    
    
    centers = torch.stack([x,y,z], dim=-1)
    
    scale_xy = uniform_range(scale_xy_range[0], scale_xy_range[1], (num_samples,)) * taper_factor
    
    if axisymmetric:
        sx = sy = scale_xy
    else:
        sx = scale_xy * uniform_range(1.0 - anisotropy, 1.0 + anisotropy, (num_samples,))
        sy = scale_xy * uniform_range(1.0 - anisotropy, 1.0 + anisotropy, (num_samples,))
        
    scale_z = uniform_range(scale_z_range[0], scale_z_range[1], (num_samples,))
    scales = torch.stack([sx, sy, scale_z], dim=-1)
    
    amplitudes = uniform_range(0.5, 1.0, (num_samples,))

    
    return (centers, scales, amplitudes)


def make_plume_volume(
    resolution: tuple[int, int, int] = (32, 32, 32),
    bounds: tuple[float, float] = (-1.0, 1.0),
    *,
    n_blobs: int = 6,
    seed: int | None = None,
    approximately_axisymmetric: bool = False,
    axis_jitter: float = 0.0,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Generate a procedural ground-truth fire-like emission volume.

    TODO ideas:
        - place blobs preferentially along the world z axis
        - generally decrease radius with height
        - perturb centres/scales so the plume is not trivial
        - if `approximately_axisymmetric`, constrain x/y offsets or construct
          the field from radial blobs
        - normalize to a convenient emission range

    Returns:
        dense volume `[D,H,W]`.
    """
    blobs = make_plume_blobs(
        n_blobs=n_blobs,
        seed=seed,
        approximately_axisymmetric=approximately_axisymmetric,
        axis_jitter=axis_jitter,
        device=device,
    )
    grid = make_world_grid(resolution, bounds=bounds, device=device)
    vol = torch.zeros(resolution, device=device)
    for blob in blobs:
        vol += gaussian_blob_field(grid, blob)
    return vol / vol.max().clamp_min(1e-8)


def make_plume_blobs(
    *,
    n_blobs: int = 6,
    seed: int | None = None,
    approximately_axisymmetric: bool = False,
    axis_jitter: float = 0.0,
    device: torch.device | str = "cpu",
) -> list[GaussianBlob]:
    """Sample the Gaussian parameters used by the procedural plume."""
    radial_spread = 0.35
    z_range = (-0.8, 0.7)
    scale_xy_range = (0.10, 0.28)
    scale_z_range = (0.15, 0.40)
    taper = 0.55
    anisotropy = 0.15
    
    
    if seed is not None:
        torch.manual_seed(seed)
        
    output = sample_blob_params(
        radial_spread=radial_spread,
        z_range=z_range,
        scale_xy_range=scale_xy_range,
        scale_z_range=scale_z_range,
        taper=taper,
        num_samples=n_blobs,
        axisymmetric=approximately_axisymmetric,
        axis_jitter=axis_jitter,
        anisotropy=anisotropy,
        device=device,
        dtype=torch.float32
    )
        
    
    centers, scales, amplitudes = output
    return [
        GaussianBlob(center, scale, float(amplitude))
        for center, scale, amplitude in zip(centers, scales, amplitudes)
    ]


def make_asymmetric_diagnostic_volume(
    resolution: tuple[int, int, int] = (32, 32, 32),
    bounds: tuple[float, float] = (-1.0, 1.0),
    *,
    seed: int | None = None,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Generate an intentionally asymmetric diagnostic volume.
    
    Three Gaussians placed at asymmetric positions to create strong x/y
    variation, making novel view divergence obvious.
    
    Returns:
        dense volume `[D,H,W]`.
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    grid = make_world_grid(resolution, bounds=bounds, device=device)
    vol = torch.zeros(resolution, device=device)
    
    # Three asymmetrically placed Gaussians
    blobs = [
        GaussianBlob(
            center_xyz=torch.tensor([-0.25, -0.4, -0.4], device=device),
            scales_xyz=torch.tensor([0.15, 0.15, 0.20], device=device),
            amplitude=1.0
        ),
        GaussianBlob(
            center_xyz=torch.tensor([0.0, 0.25, 0.0], device=device),
            scales_xyz=torch.tensor([0.12, 0.12, 0.18], device=device),
            amplitude=0.8
        ),
        GaussianBlob(
            center_xyz=torch.tensor([0.2, -0.1, 0.5], device=device),
            scales_xyz=torch.tensor([0.10, 0.10, 0.15], device=device),
            amplitude=0.9
        ),
    ]
    
    for blob in blobs:
        eval_blob = gaussian_blob_field(grid, blob)
        vol += eval_blob
    
    vol = vol / vol.max().clamp_min(1e-8)
    return vol
        
        
        
        
       
