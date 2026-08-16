from __future__ import annotations

from torch import Tensor
import torch

def image_l1(predicted: Tensor, target: Tensor) -> Tensor:
    """Mean absolute image error."""
    return torch.mean(torch.abs(predicted - target))


def normalized_volume_l1(predicted: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """`||pred-target||_1 / (||target||_1 + eps)` for synthetic GT tests."""
    return (torch.sum(torch.abs(predicted - target)))/(torch.sum(torch.abs(target)) + eps)


def relative_mass_error(predicted: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """Optional diagnostic comparing integrated emission magnitudes."""
    return (torch.abs(torch.sum(predicted) - torch.sum(target)))/(torch.sum(torch.abs(target)) + eps)