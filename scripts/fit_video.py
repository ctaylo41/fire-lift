"""Fit an axisymmetric volume sequence to fixed-camera fire footage."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

from firelift.fit.reconstruct import FitConfig, LossWeights
from firelift.fit.sequence import fit_sequence
from firelift.io.video import load_video_frames, rgb_to_emission_target
from firelift.render.camera import OrthographicCamera
from firelift.render.raymarch import render_emission
from firelift.volume.axisymmetric import AxisymmetricVolume


def _to_uint8(image: torch.Tensor) -> np.ndarray:
    values = image.detach().cpu().numpy()
    values = (values - values.min()) / max(values.max() - values.min(), 1e-8)
    return (values * 255.0).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/video_fit"))
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n-samples", type=int, default=48)
    parser.add_argument("--profile-height", type=int, default=32)
    parser.add_argument("--profile-radius", type=int, default=16)
    parser.add_argument("--temporal-weight", type=float, default=0.01)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="Compute device; auto selects CUDA, then Apple Metal, then CPU.",
    )
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this PyTorch environment")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available in this PyTorch environment")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    rgb_frames = load_video_frames(args.video, max_frames=args.max_frames)
    if not rgb_frames:
        raise RuntimeError(f"No frames decoded from {args.video}")
    frames = [rgb_to_emission_target(frame).to(device) for frame in rgb_frames]
    height, width = frames[0].shape

    camera = OrthographicCamera.look_at(
        eye=torch.tensor([0.0, -2.5, 0.0], device=device),
        target=torch.zeros(3, device=device),
        up=torch.tensor([0.0, 0.0, 1.0], device=device),
        ortho_width=2.2,
        near=1.5,
        far=3.5,
    )
    config = FitConfig(
        steps=args.steps,
        lr=args.lr,
        n_samples_per_ray=args.n_samples,
        log_every=max(1, args.steps // 4),
    )
    result = fit_sequence(
        frames,
        camera,
        lambda: AxisymmetricVolume(
            profile_resolution=(args.profile_height, args.profile_radius)
        ).to(device),
        warm_start=True,
        temporal_weight=args.temporal_weight,
        config=config,
        weights=LossWeights(image=1.0),
    )

    stats_path = args.output / "stats.csv"
    comparison_frames: list[np.ndarray] = []
    with stats_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame",
                "height",
                "width",
                "target_mean",
                "target_max",
                "final_loss",
                "final_image_loss",
                "final_temporal_loss",
                "volume_mean",
                "volume_max",
            ],
        )
        writer.writeheader()
        for index, (frame, volume, history) in enumerate(
            zip(frames, result.volumes, result.histories)
        ):
            with torch.no_grad():
                prediction = render_emission(
                    volume, camera, height, width, n_samples=args.n_samples
                )
                materialized = volume.materialize((32, 32, 32))
            target_image = _to_uint8(frame)
            prediction_image = _to_uint8(prediction)
            iio.imwrite(args.output / f"target_{index:04d}.png", target_image)
            iio.imwrite(args.output / f"prediction_{index:04d}.png", prediction_image)
            comparison_frames.append(
                np.concatenate([target_image, prediction_image], axis=1)
            )
            writer.writerow(
                {
                    "frame": index,
                    "height": height,
                    "width": width,
                    "target_mean": float(frame.mean()),
                    "target_max": float(frame.max()),
                    "final_loss": history["total"][-1],
                    "final_image_loss": history["image"][-1],
                    "final_temporal_loss": history["temporal"][-1],
                    "volume_mean": float(materialized.mean()),
                    "volume_max": float(materialized.max()),
                }
            )

    iio.imwrite(
        args.output / "fit_comparison.mp4",
        np.stack(comparison_frames),
        fps=8,
    )

    print(f"video={args.video}")
    print(f"device={device}")
    print(f"frames={len(frames)} resolution={height}x{width}")
    print(f"stats={stats_path}")
    print(f"previews={args.output / 'prediction_*.png'}")
    print(f"comparison={args.output / 'fit_comparison.mp4'}")


if __name__ == "__main__":
    main()
