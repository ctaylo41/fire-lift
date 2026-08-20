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
    frame_weights: Sequence[Tensor] | None = None,
    warm_start: bool = True,
    temporal_weight: float = 0.0,
    config: FitConfig | None = None,
    weights: LossWeights | None = None,
    adaptive: bool = False,
    first_frame_steps: int | None = None,
    later_frame_steps: int | None = None,
    adaptive_chunk_steps: int = 250,
    adaptive_improvement_threshold: float = 1e-3,
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
    if adaptive_chunk_steps <= 0:
        raise ValueError("adaptive_chunk_steps must be positive")
    if adaptive_improvement_threshold < 0.0:
        raise ValueError("adaptive_improvement_threshold must be non-negative")

    height, width = frames[0].shape
    if frames[0].ndim != 2:
        raise ValueError("Each frame must be a scalar image with shape [H, W]")
    if any(frame.ndim != 2 or frame.shape != (height, width) for frame in frames):
        raise ValueError("All frames must have the same [H, W] shape")
    if frame_weights is not None:
        if len(frame_weights) != len(frames):
            raise ValueError("frame_weights must have the same length as frames")
        if any(weight.shape != (height, width) for weight in frame_weights):
            raise ValueError("Each frame weight map must have shape [H, W]")

    fit_config = config if config is not None else FitConfig()
    fit_weights = weights if weights is not None else LossWeights()
    volumes: list[VolumeField] = []
    histories: list[dict[str, list[float]]] = []
    previous_volume: VolumeField | None = None

    for frame_index, frame in enumerate(frames):
        print(f"frame={frame_index + 1}/{len(frames)} start")
        volume = volume_factory()
        if warm_start and previous_volume is not None:
            volume.load_state_dict(previous_volume.state_dict())

        temporal_target = None
        if previous_volume is not None and temporal_weight > 0.0:
            temporal_target = previous_volume.regularization_field().detach().clone()

        weight = None if frame_weights is None else frame_weights[frame_index]
        observation = [Observation(frame, camera, weight=weight)]

        if not adaptive:
            history = fit_volume(
                volume,
                observation,
                weights=fit_weights,
                config=fit_config,
                temporal_target=temporal_target,
                temporal_weight=temporal_weight,
            )
        else:
            maximum_steps = (
                first_frame_steps if frame_index == 0 else later_frame_steps
            )
            maximum_steps = fit_config.steps if maximum_steps is None else maximum_steps
            if maximum_steps <= 0:
                raise ValueError("adaptive step budgets must be positive")

            history = {key: [] for key in ("total", "image", "sparsity", "variation", "temporal")}
            steps_done = 0
            while steps_done < maximum_steps:
                chunk_steps = (
                    maximum_steps
                    if frame_index == 0
                    else min(adaptive_chunk_steps, maximum_steps - steps_done)
                )
                chunk_config = FitConfig(
                    steps=chunk_steps,
                    lr=fit_config.lr,
                    n_samples_per_ray=fit_config.n_samples_per_ray,
                    log_every=fit_config.log_every,
                )
                chunk_history = fit_volume(
                    volume,
                    observation,
                    weights=fit_weights,
                    config=chunk_config,
                    temporal_target=temporal_target,
                    temporal_weight=temporal_weight,
                )
                for key in history:
                    history[key].extend(chunk_history[key])
                steps_done += chunk_steps

                if frame_index == 0 or steps_done >= maximum_steps:
                    break
                start_loss = chunk_history["total"][0]
                end_loss = chunk_history["total"][-1]
                relative_improvement = (start_loss - end_loss) / max(abs(start_loss), 1e-8)
                if relative_improvement <= adaptive_improvement_threshold:
                    print(
                        f"adaptive_stop frame={frame_index} steps={steps_done} "
                        f"relative_improvement={relative_improvement:.6f}"
                    )
                    break
        volumes.append(volume)
        histories.append(history)
        previous_volume = volume
        print(
            f"frame={frame_index + 1}/{len(frames)} done "
            f"steps={len(history['total'])} "
            f"final_loss={history['total'][-1]:.6f}"
        )

    return SequenceFitResult(volumes=volumes, histories=histories)
        
