from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from torch import Tensor

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
) -> SequenceFitResult:
    """Fit one volume per frame of a fixed-camera video.

    Intentionally left as a later Part-1 task.

    Design questions for you to decide:
        - warm-start theta_t from theta_{t-1}?
        - optimize each frame sequentially or jointly?
        - apply temporal prior in profile space or materialized 3D space?
        - how many optimizer steps are needed after warm-starting?
    """
    raise NotImplementedError
