from __future__ import annotations

import torch
from torch import Tensor


def sparsity_l1(field: Tensor) -> Tensor:
    """Mean absolute emission magnitude for a non-negative field."""
    return field.abs().mean()


def total_variation(field: Tensor) -> Tensor:
    """Anisotropic total variation for a 2D or 3D scalar field."""
    tv = field.new_zeros(())
    for axis in range(field.ndim):
        if field.shape[axis] > 1:
            tv = tv + torch.diff(field, dim=axis).abs().sum()
    return tv


def temporal_l1(current: Tensor, previous: Tensor) -> Tensor:
    """Simple temporal consistency prior between consecutive fields."""
    return torch.abs(current - previous).mean()
