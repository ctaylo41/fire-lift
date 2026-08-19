from __future__ import annotations

import torch
from torch import Tensor

from .base import VolumeField
from firelift.synth.generate import GaussianBlob, gaussian_blob_field


class AnalyticGaussianVolume(VolumeField):
    """Fixed continuous field formed by summing Gaussian emission blobs."""

    def __init__(self, blobs: list[GaussianBlob], normalization: float = 1.0) -> None:
        super().__init__()
        self.blobs = blobs
        self.normalization = normalization

    def sample(self, points_xyz: Tensor) -> Tensor:
        values = torch.zeros(points_xyz.shape[:-1], device=points_xyz.device, dtype=points_xyz.dtype)
        for blob in self.blobs:
            values = values + gaussian_blob_field(points_xyz, blob)
        return values / self.normalization

    def regularization_field(self) -> Tensor:
        return torch.zeros(1, dtype=torch.float32)