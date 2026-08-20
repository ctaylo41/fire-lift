from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import imageio.v3 as iio
import torch
import torch.nn.functional as F
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
    for frame in iio.imiter(path):
        frames.append(_to_rgb_tensor(frame))
        if max_frames is not None and len(frames) >= max_frames:
            break
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


def srgb_to_linear_rgb(frame_rgb: Tensor) -> Tensor:
    """Approximate inverse sRGB transfer for values in [0, 1]."""
    threshold = 0.04045
    low = frame_rgb / 12.92
    high = torch.pow((frame_rgb + 0.055) / 1.055, 2.4)
    return torch.where(frame_rgb <= threshold, low, high)


def linear_rgb_to_srgb(frame_rgb_linear: Tensor) -> Tensor:
    """Approximate forward sRGB transfer for values in [0, 1]."""
    threshold = 0.0031308
    low = frame_rgb_linear * 12.92
    high = 1.055 * torch.pow(frame_rgb_linear.clamp_min(0.0), 1.0 / 2.4) - 0.055
    return torch.where(frame_rgb_linear <= threshold, low, high)


def linear_rgb_to_luminance(frame_rgb_linear: Tensor) -> Tensor:
    """Convert linear RGB to linear luminance using Rec.709 coefficients."""
    coeffs = frame_rgb_linear.new_tensor([0.2126, 0.7152, 0.0722])
    return (frame_rgb_linear * coeffs).sum(dim=-1)


def resize_frames_preserve_aspect(frames: Sequence[Tensor], target_long_side: int) -> list[Tensor]:
    """Resize all frames once using bilinear filtering while preserving aspect."""
    if target_long_side <= 0:
        raise ValueError("target_long_side must be positive")
    if not frames:
        return []

    height, width = frames[0].shape[:2]
    scale = target_long_side / float(max(height, width))
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))

    stacked = torch.stack([frame.to(torch.float32) for frame in frames], dim=0)
    resized = F.interpolate(
        stacked.permute(0, 3, 1, 2),
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
    )
    resized = resized.permute(0, 2, 3, 1)
    return [resized[i].contiguous() for i in range(resized.shape[0])]


def preprocess_fire_frames(
    frames_rgb: Sequence[Tensor],
    *,
    crop: tuple[int, int, int, int] | None = None,
    target_long_side: int | None = None,
    linearize: bool = True,
    background_subtract: bool = True,
    background_border_fraction: float = 0.12,
    background_percentile: float = 50.0,
    saturation_threshold: float = 0.98,
) -> dict[str, object]:
    """Apply a disciplined preprocessing pass for real fire footage."""
    if not frames_rgb:
        raise ValueError("frames_rgb must contain at least one frame")
    if not 0.0 < background_border_fraction < 0.5:
        raise ValueError("background_border_fraction must be in (0, 0.5)")
    if not 0.0 <= background_percentile <= 100.0:
        raise ValueError("background_percentile must be in [0, 100]")
    if not 0.0 <= saturation_threshold <= 1.0:
        raise ValueError("saturation_threshold must be in [0, 1]")

    working: list[Tensor] = [frame.to(torch.float32).clamp(0.0, 1.0) for frame in frames_rgb]
    if crop is not None:
        working = crop_frames(working, crop)
    if target_long_side is not None:
        working = resize_frames_preserve_aspect(working, target_long_side)

    observed_rgb = torch.stack(working, dim=0)
    saturation_mask = observed_rgb.max(dim=-1).values >= saturation_threshold

    if linearize:
        linear_rgb = srgb_to_linear_rgb(observed_rgb)
    else:
        linear_rgb = observed_rgb

    background_rgb = linear_rgb.new_zeros(3)
    if background_subtract:
        _, height, width, _ = linear_rgb.shape
        by = max(1, int(round(height * background_border_fraction)))
        bx = max(1, int(round(width * background_border_fraction)))

        border = torch.zeros((height, width), dtype=torch.bool, device=linear_rgb.device)
        border[:by, :] = True
        border[-by:, :] = True
        border[:, :bx] = True
        border[:, -bx:] = True

        border_pixels = linear_rgb[:, border, :].reshape(-1, 3)
        background_rgb = torch.quantile(border_pixels, q=background_percentile / 100.0, dim=0)
        linear_rgb = (linear_rgb - background_rgb.view(1, 1, 1, 3)).clamp_min(0.0)

    luminance = linear_rgb_to_luminance(linear_rgb)
    return {
        "observed_rgb": [observed_rgb[i] for i in range(observed_rgb.shape[0])],
        "linear_rgb": [linear_rgb[i] for i in range(linear_rgb.shape[0])],
        "luminance": [luminance[i] for i in range(luminance.shape[0])],
        "saturation_mask": [saturation_mask[i] for i in range(saturation_mask.shape[0])],
        "background_rgb": background_rgb,
    }
