from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor


def load_video_frames(path: str | Path, *, max_frames: int | None = None) -> list[Tensor]:
    """Decode a video to float RGB tensors `[H,W,3]` in [0,1].

    TODO:
        Use imageio or another simple decoder. Do not silently resize/crop here.
    """
    raise NotImplementedError


def rgb_to_emission_target(frame_rgb: Tensor) -> Tensor:
    """Convert one controlled fire frame into a scalar fitting target `[H,W]`.

    First version can be deliberately simple (e.g. luminance / max RGB) but
    document the assumption. Real-camera response is not truly linear.
    """
    raise NotImplementedError


def crop_frames(frames: Iterable[Tensor], box_xyxy: tuple[int, int, int, int]) -> list[Tensor]:
    """Apply one fixed crop to every frame."""
    raise NotImplementedError
