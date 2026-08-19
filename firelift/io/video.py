from __future__ import annotations

from pathlib import Path
from typing import Iterable

import imageio.v3 as iio
import torch
from torch import Tensor


def _to_rgb_tensor(frame) -> Tensor:
    """Convert one decoded frame to a float32 RGB tensor without resizing."""
    tensor = torch.as_tensor(frame)
    if tensor.dtype == torch.uint8:
        tensor = tensor.to(torch.float32) / 255.0
    elif tensor.dtype == torch.uint16:
        tensor = tensor.to(torch.float32) / 65535.0
    elif tensor.is_floating_point():
        tensor = tensor.to(torch.float32)
    return tensor.contiguous()

def load_video_frames(path: str | Path, *, max_frames: int | None = None) -> list[Tensor]:
    """Decode a video to float RGB tensors `[H,W,3]` in [0,1].

    Frames are decoded lazily, so a frame limit does not require loading the
    complete video. Grayscale frames are expanded to RGB and alpha channels
    are discarded; spatial dimensions are never changed.
    """
    frames: list[Tensor] = []
    for index, frame in enumerate(iio.imiter(path)):
        if max_frames is not None and index >= max_frames:
            break
        frames.append(_to_rgb_tensor(frame))
    return frames


def rgb_to_emission_target(frame_rgb: Tensor) -> Tensor:
    """Convert one controlled fire frame into a scalar fitting target `[H,W]`.

    First version can be deliberately simple (e.g. luminance / max RGB) but
    document the assumption. Real-camera response is not truly linear.
    """
    # Use the brightest channel as a simple linear emission proxy.
    return frame_rgb.max(dim=-1).values


def crop_frames(frames: Iterable[Tensor], box_xyxy: tuple[int, int, int, int]) -> list[Tensor]:
    """Apply one fixed crop to every frame."""
    x0, y0, x1, y1 = box_xyxy
    cropped: list[Tensor] = []
    for frame in frames:
        height, width = frame.shape[:2]
        cropped.append(frame[y0:y1, x0:x1].clone())
    return cropped
