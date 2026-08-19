from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from firelift.render.camera import OrthographicCamera
from firelift.render.raymarch import render_emission
from firelift.volume.base import VolumeField
from firelift.volume.priors import sparsity_l1, total_variation


@dataclass
class Observation:
    image: Tensor                    # [H,W]
    camera: OrthographicCamera


@dataclass
class LossWeights:
    image: float = 1.0
    sparsity: float = 0.0
    tv: float = 0.0


@dataclass
class FitConfig:
    steps: int = 1000
    lr: float = 1e-2
    n_samples_per_ray: int = 96
    log_every: int = 50


def image_loss(predicted: Tensor, target: Tensor) -> Tensor:
    """First baseline image-space reconstruction loss.

    TODO: start with L1. Keep this as a function so alternatives are easy.
    """
    return torch.abs(target - predicted).mean()


def compute_loss(
    volume: VolumeField,
    observations: Sequence[Observation],
    *,
    weights: LossWeights,
    n_samples_per_ray: int,
    temporal_target: Tensor | None = None,
    temporal_weight: float = 0.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    device = volume.regularization_field().device
    dtype = volume.regularization_field().dtype
    
    total_loss = torch.tensor(0.0, device=device, dtype=dtype)
    
    loss_components = {
        "image" : torch.tensor(0.0, device=device, dtype=dtype),
        "sparsity" : torch.tensor(0.0, device=device, dtype=dtype),
        "variation" : torch.tensor(0.0, device=device, dtype=dtype),
    }
    
    for observation in observations:
        image = observation.image
        camera = observation.camera
        height, width = image.shape
        render = render_emission(volume, camera, height, width, n_samples=n_samples_per_ray)
        
        loss = image_loss(render, image)
        
        total_loss += loss*weights.image
        
        loss_components["image"] += loss.detach()
        
    total_loss /= len(observations)
    loss_components["image"] /= len(observations)
    
    if weights.sparsity != 0.0:     
        sparsity_loss = sparsity_l1(volume.regularization_field())
        sparsity_loss *= weights.sparsity
        total_loss += sparsity_loss
        loss_components["sparsity"] = sparsity_loss.detach()
    
    
    if weights.tv != 0.0:
        variation_loss = total_variation(volume.regularization_field())
        variation_loss *= weights.tv
        total_loss += variation_loss
        loss_components["variation"] = variation_loss.detach()

    if temporal_target is not None:
        current_field = volume.regularization_field()
        if current_field.shape != temporal_target.shape:
            raise ValueError(
                "Temporal target shape must match the current regularization field: "
                f"{tuple(temporal_target.shape)} != {tuple(current_field.shape)}"
            )
        temporal_loss = torch.mean(torch.abs(current_field - temporal_target.detach()))
        total_loss += temporal_weight * temporal_loss
        loss_components["temporal"] = (temporal_weight * temporal_loss).detach()
        
    return (total_loss, loss_components)
    

def fit_volume(
    volume: VolumeField,
    observations: Sequence[Observation],
    *,
    weights: LossWeights,
    config: FitConfig,
    temporal_target: Tensor | None = None,
    temporal_weight: float = 0.0,
) -> dict[str, list[float]]:
    """Optimize volume parameters with Adam.

    TODO:
        - construct Adam over `volume.parameters()`
        - zero grad -> compute_loss -> backward -> step
        - collect history
        - print/log every `log_every`
        - do not put experiment-specific plotting here

    Returns:
        history dict such as `{"total": [...], "image": [...], ...}`.
    """
    history_tracking = {
        "total": [],
        "image": [],
        "sparsity": [],
        "variation": [],
        "temporal": [],
    }

    optimizer = torch.optim.Adam(volume.parameters(), lr=config.lr)

    for step in range(config.steps):
        optimizer.zero_grad()

        total_loss, component_loss = compute_loss(
            volume,
            observations,
            weights=weights,
            n_samples_per_ray=config.n_samples_per_ray,
            temporal_target=temporal_target,
            temporal_weight=temporal_weight,
        )

        total_loss.backward()
        optimizer.step()

        history_tracking["total"].append(float(total_loss.item()))
        history_tracking["sparsity"].append(float(component_loss["sparsity"].item()))
        history_tracking["image"].append(float(component_loss["image"].item()))
        history_tracking["variation"].append(float(component_loss["variation"].item()))
        history_tracking["temporal"].append(float(component_loss.get("temporal", torch.tensor(0.0)).item()))

        if step % config.log_every == 0 or step == config.steps - 1:
            print(
                f"epoch={step + 1} step={step} "
                f"total_loss={total_loss.item():.6f} "
                f"image_loss={component_loss['image'].item():.6f} "
                f"sparsity_loss={component_loss['sparsity'].item():.6f} "
                f"variation_loss={component_loss['variation'].item():.6f} "
                f"temporal_loss={component_loss.get('temporal', torch.tensor(0.0)).item():.6f}"
            )

    return history_tracking
