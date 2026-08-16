from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor


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


def make_plume_volume(
    resolution: tuple[int, int, int] = (32, 32, 32),
    *,
    n_blobs: int = 6,
    seed: int | None = None,
    approximately_axisymmetric: bool = False,
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
    if seed is not None:
        torch.manual_seed(seed)
        
    grid = make_world_grid(resolution, device=device)
        
    vol = torch.zeros(resolution, device=device)
    
    z_bias = 0.2
    for _ in range(n_blobs):
        pos = torch.randn(3, device=device)
        pos[2]+=z_bias
        
        if approximately_axisymmetric:
            pos[0] = torch.randn(1, device=device) * 0.1
            pos[1] = torch.randn(1, device=device) * 0.1
        
        scale = torch.rand(3, device=device) / (1.0 + pos[2])
        
        amplitude = torch.rand(1, device=device).item()
                
        blob = GaussianBlob(pos, scale, amplitude)
        
        eval_blob = gaussian_blob_field(grid, blob)
        
        vol += eval_blob
        
    vol = vol/vol.max()
    vol = vol.to(device)
    return vol
        
        
        
        
       
