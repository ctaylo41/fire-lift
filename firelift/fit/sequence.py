from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor

from firelift.fit.reconstruct import FitConfig, LossWeights, Observation, fit_volume
from firelift.render.camera import OrthographicCamera
from firelift.volume.base import VolumeField


@dataclass
class SequenceFitResult:
    volumes: list[VolumeField]
    histories: list[dict[str, list[float]]]


def fit_sequence(
    frames: Sequence[Tensor],
    camera: OrthographicCamera,
    volume_factory: Callable[[], VolumeField],
    *,
    warm_start: bool = True,
    temporal_weight: float = 0.0,
    config: FitConfig | None = None,
    weights: LossWeights | None = None,
) -> SequenceFitResult:
    """Fit one volume per frame of a fixed-camera video.

    Frames are fitted sequentially. When enabled, warm-starting copies the
    previous volume parameters into the next volume, and the temporal prior
    acts on each volume's native regularization field.
    """
    if not frames:
        raise ValueError("frames must contain at least one frame")
    if temporal_weight < 0.0:
        raise ValueError("temporal_weight must be non-negative")

    height, width = frames[0].shape
    if frames[0].ndim != 2:
        raise ValueError("Each frame must be a scalar image with shape [H, W]")
    if any(frame.ndim != 2 or frame.shape != (height, width) for frame in frames):
        raise ValueError("All frames must have the same [H, W] shape")

    fit_config = config if config is not None else FitConfig()
    fit_weights = weights if weights is not None else LossWeights()
    volumes: list[VolumeField] = []
    histories: list[dict[str, list[float]]] = []
    previous_volume: VolumeField | None = None

    for frame in frames:
        volume = volume_factory()
        if warm_start and previous_volume is not None:
            volume.load_state_dict(previous_volume.state_dict())

        temporal_target = None
        if previous_volume is not None and temporal_weight > 0.0:
            temporal_target = previous_volume.regularization_field().detach().clone()

        history = fit_volume(
            volume,
            [Observation(frame, camera)],
            weights=fit_weights,
            config=fit_config,
            temporal_target=temporal_target,
            temporal_weight=temporal_weight,
        )
        volumes.append(volume)
        histories.append(history)
        previous_volume = volume

    return SequenceFitResult(volumes=volumes, histories=histories)
        
