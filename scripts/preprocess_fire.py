"""Preprocess real fire footage for reconstruction on linearized intensity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

from firelift.io.video import (
    linear_rgb_to_srgb,
    load_video_frames,
    preprocess_fire_frames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/preprocessed_fire"))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--crop", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), required=True)
    parser.add_argument("--target-long-side", type=int, default=128)
    parser.add_argument("--saturation-threshold", type=float, default=0.98)
    parser.add_argument("--background-border-fraction", type=float, default=0.12)
    parser.add_argument("--background-percentile", type=float, default=50.0)
    parser.add_argument("--save-preview-video", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def write_preview_pngs(output_dir: Path, rgb_linear: torch.Tensor, luminance: torch.Tensor, saturation_mask: torch.Tensor) -> None:
    rgb_display = linear_rgb_to_srgb(rgb_linear).clamp(0.0, 1.0)
    rgb_u8 = (rgb_display.cpu().numpy() * 255.0).astype(np.uint8)

    lum = luminance.cpu().numpy()
    lum = lum / max(float(lum.max()), 1e-8)
    lum_u8 = (lum * 255.0).astype(np.uint8)

    sat_u8 = (saturation_mask.cpu().numpy().astype(np.uint8) * 255)

    for index in range(rgb_u8.shape[0]):
        iio.imwrite(output_dir / f"rgb_linear_{index:04d}.png", rgb_u8[index])
        iio.imwrite(output_dir / f"luminance_{index:04d}.png", lum_u8[index])
        iio.imwrite(output_dir / f"saturation_mask_{index:04d}.png", sat_u8[index])


def write_preview_video(output_dir: Path, rgb_linear: torch.Tensor) -> None:
    rgb_display = linear_rgb_to_srgb(rgb_linear).clamp(0.0, 1.0)
    rgb_u8 = (rgb_display.cpu().numpy() * 255.0).astype(np.uint8)
    iio.imwrite(output_dir / "preview_linear_rgb.mp4", rgb_u8, fps=8)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rgb_frames = load_video_frames(args.video, max_frames=args.max_frames)
    if not rgb_frames:
        raise RuntimeError(f"No frames decoded from {args.video}")

    preprocessed = preprocess_fire_frames(
        rgb_frames,
        crop=tuple(args.crop),
        target_long_side=args.target_long_side,
        linearize=True,
        background_subtract=True,
        background_border_fraction=args.background_border_fraction,
        background_percentile=args.background_percentile,
        saturation_threshold=args.saturation_threshold,
    )

    rgb_linear_bg = torch.stack(preprocessed["linear_rgb"], dim=0)
    lum_linear = torch.stack(preprocessed["luminance"], dim=0)
    saturation_mask = torch.stack(preprocessed["saturation_mask"], dim=0)
    background = preprocessed["background_rgb"]

    np.savez_compressed(
        args.output / "preprocessed_fire.npz",
        rgb_linear=rgb_linear_bg.cpu().numpy(),
        luminance=lum_linear.cpu().numpy(),
        saturation_mask=saturation_mask.cpu().numpy().astype(np.uint8),
    )

    metadata = {
        "video": str(args.video),
        "num_frames": int(rgb_linear_bg.shape[0]),
        "crop_xyxy": list(map(int, args.crop)),
        "target_long_side": int(args.target_long_side),
        "final_height": int(rgb_linear_bg.shape[1]),
        "final_width": int(rgb_linear_bg.shape[2]),
        "background_linear_rgb": [float(v) for v in background.cpu().tolist()],
        "saturation_threshold": float(args.saturation_threshold),
        "background_border_fraction": float(args.background_border_fraction),
        "background_percentile": float(args.background_percentile),
    }
    with (args.output / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    write_preview_pngs(args.output, rgb_linear_bg, lum_linear, saturation_mask)
    if args.save_preview_video:
        write_preview_video(args.output, rgb_linear_bg)

    print(f"video={args.video}")
    print(f"frames={metadata['num_frames']}")
    print(f"crop={metadata['crop_xyxy']}")
    print(f"size={metadata['final_width']}x{metadata['final_height']}")
    print(f"background_linear_rgb={metadata['background_linear_rgb']}")
    print(f"saved={args.output / 'preprocessed_fire.npz'}")


if __name__ == "__main__":
    main()