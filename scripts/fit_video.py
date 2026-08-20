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
from firelift.io.video import linear_rgb_to_srgb, load_video_frames, preprocess_fire_frames
from firelift.render.camera import OrthographicCamera
from firelift.render.raymarch import render_emission
from firelift.volume.axisymmetric import AxisymmetricVolume, BentAxisymmetricVolume, FourierVolume


def _to_uint8(image: torch.Tensor) -> np.ndarray:
    values = image.detach().cpu().numpy()
    values = (values - values.min()) / max(values.max() - values.min(), 1e-8)
    return (values * 255.0).astype(np.uint8)


def _rgb_to_uint8(image: torch.Tensor) -> np.ndarray:
    values = image.detach().cpu().clamp(0.0, 1.0).numpy()
    return (values * 255.0).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/video_fit"))
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--frame-index", type=int, default=None)
    parser.add_argument("--crop", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), default=None)
    parser.add_argument(
        "--target-long-side",
        type=int,
        default=None,
        help="Resize each cropped frame once so its longest side matches this value.",
    )
    parser.add_argument(
        "--linearize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Convert observed sRGB video frames to approximate linear RGB before fitting.",
    )
    parser.add_argument(
        "--background-subtract",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Estimate dark-level background from frame borders and subtract conservatively.",
    )
    parser.add_argument("--background-border-fraction", type=float, default=0.12)
    parser.add_argument("--background-percentile", type=float, default=50.0)
    parser.add_argument("--saturation-threshold", type=float, default=0.98)
    parser.add_argument(
        "--saturation-weight",
        type=float,
        default=0.25,
        help="Relative loss weight assigned to saturated pixels.",
    )
    parser.add_argument(
        "--save-preprocessed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save preprocessed linear RGB/luminance/mask artifacts in output directory.",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--adaptive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adaptive-chunk", type=int, default=250)
    parser.add_argument("--adaptive-threshold", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n-samples", type=int, default=48)
    parser.add_argument("--profile-height", type=int, default=32)
    parser.add_argument("--profile-radius", type=int, default=16)
    parser.add_argument("--centerline-points", type=int, default=6)
    parser.add_argument("--temporal-weight", type=float, default=0.01)
    parser.add_argument("--ortho-width", type=float, default=0.8)
    parser.add_argument("--max-radius", type=float, default=0.35)
    parser.add_argument("--z-min", type=float, default=-0.8)
    parser.add_argument("--z-max", type=float, default=0.8)
    parser.add_argument("--init-value", type=float, default=-3.0)
    parser.add_argument(
        "--representation",
        choices=("axisymmetric", "bent_axisymmetric", "fourier1"),
        default="axisymmetric",
    )
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

    if args.frame_index is not None:
        if args.frame_index < 0:
            raise ValueError("frame-index must be non-negative")
        decoded = load_video_frames(args.video, max_frames=args.frame_index + 1)
        if len(decoded) <= args.frame_index:
            raise RuntimeError(f"Video has no frame at index {args.frame_index}")
        rgb_frames = [decoded[args.frame_index]]
    else:
        rgb_frames = load_video_frames(args.video, max_frames=args.max_frames)
    if not rgb_frames:
        raise RuntimeError(f"No frames decoded from {args.video}")
    print(f"decoded_frames={len(rgb_frames)} requested_max_frames={args.max_frames}")

    preprocessed = preprocess_fire_frames(
        rgb_frames,
        crop=tuple(args.crop) if args.crop is not None else None,
        target_long_side=args.target_long_side,
        linearize=args.linearize,
        background_subtract=args.background_subtract,
        background_border_fraction=args.background_border_fraction,
        background_percentile=args.background_percentile,
        saturation_threshold=args.saturation_threshold,
    )
    frames = [frame.to(device) for frame in preprocessed["luminance"]]
    saturation_masks = [mask.to(device) for mask in preprocessed["saturation_mask"]]
    if not 0.0 < args.saturation_weight <= 1.0:
        raise ValueError("saturation-weight must be in (0, 1]")
    frame_weights = [
        torch.where(mask, torch.full_like(mask, args.saturation_weight, dtype=torch.float32), torch.ones_like(mask, dtype=torch.float32))
        for mask in saturation_masks
    ]

    height, width = frames[0].shape

    camera = OrthographicCamera.look_at(
        eye=torch.tensor([0.0, -2.5, 0.0], device=device),
        target=torch.zeros(3, device=device),
        up=torch.tensor([0.0, 0.0, 1.0], device=device),
        ortho_width=args.ortho_width,
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
        lambda: (
            FourierVolume(
                profile_resolution=(args.profile_height, args.profile_radius),
                max_radius=args.max_radius,
                z_bounds=(args.z_min, args.z_max),
                init_value=args.init_value,
            )
            if args.representation == "fourier1"
            else BentAxisymmetricVolume(
                profile_resolution=(args.profile_height, args.profile_radius),
                max_radius=args.max_radius,
                z_bounds=(args.z_min, args.z_max),
                init_value=args.init_value,
                centerline_points=args.centerline_points,
            )
            if args.representation == "bent_axisymmetric"
            else AxisymmetricVolume(
                profile_resolution=(args.profile_height, args.profile_radius),
                max_radius=args.max_radius,
                z_bounds=(args.z_min, args.z_max),
                init_value=args.init_value,
            )
        ).to(device),
        frame_weights=frame_weights,
        warm_start=True,
        temporal_weight=0.0 if len(frames) == 1 else args.temporal_weight,
        config=config,
        weights=LossWeights(image=1.0),
        adaptive=args.adaptive,
        first_frame_steps=args.steps,
        later_frame_steps=args.steps,
        adaptive_chunk_steps=args.adaptive_chunk,
        adaptive_improvement_threshold=args.adaptive_threshold,
    )

    stats_path = args.output / "stats.csv"
    profile_arrays: list[np.ndarray] = []
    centerline_arrays: list[np.ndarray] = []
    comparison_frames: list[np.ndarray] = []
    if args.save_preprocessed:
        linear_rgb = torch.stack(preprocessed["linear_rgb"], dim=0)
        sat_masks = torch.stack(preprocessed["saturation_mask"], dim=0)
        np.savez_compressed(
            args.output / "preprocessed_fire.npz",
            rgb_linear=linear_rgb.cpu().numpy(),
            luminance=torch.stack(preprocessed["luminance"], dim=0).cpu().numpy(),
            saturation_mask=sat_masks.cpu().numpy().astype(np.uint8),
        )
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
                profile_arrays.append(volume.theta.detach().cpu().numpy().copy())
                if hasattr(volume, "centerline"):
                    centerline_arrays.append(volume.centerline.detach().cpu().numpy().copy())
            target_image = _to_uint8(frame)
            prediction_image = _to_uint8(prediction)
            iio.imwrite(args.output / f"target_{index:04d}.png", target_image)
            iio.imwrite(args.output / f"prediction_{index:04d}.png", prediction_image)
            if args.save_preprocessed:
                rgb_linear = preprocessed["linear_rgb"][index]
                rgb_preview = _rgb_to_uint8(linear_rgb_to_srgb(rgb_linear))
                sat_preview = (preprocessed["saturation_mask"][index].cpu().numpy().astype(np.uint8) * 255)
                iio.imwrite(args.output / f"rgb_linear_{index:04d}.png", rgb_preview)
                iio.imwrite(args.output / f"saturation_mask_{index:04d}.png", sat_preview)
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
    if args.save_preprocessed:
        preview_frames = [
            _rgb_to_uint8(linear_rgb_to_srgb(rgb)) for rgb in preprocessed["linear_rgb"]
        ]
        iio.imwrite(args.output / "preprocessed_linear_rgb.mp4", np.stack(preview_frames), fps=8)
    checkpoint = {
        "theta": np.stack(profile_arrays),
        "representation": args.representation,
        "profile_height": args.profile_height,
        "profile_radius": args.profile_radius,
        "centerline_points": args.centerline_points,
        "max_radius": args.max_radius,
        "z_min": args.z_min,
        "z_max": args.z_max,
    }
    if centerline_arrays:
        checkpoint["centerline"] = np.stack(centerline_arrays)
    np.savez_compressed(args.output / "profiles.npz", **checkpoint)

    print(f"video={args.video}")
    print(f"device={device}")
    print(f"representation={args.representation}")
    print(f"frames={len(frames)} resolution={height}x{width}")
    print(f"linearize={args.linearize} background_subtract={args.background_subtract}")
    print(f"saturation_threshold={args.saturation_threshold} saturation_weight={args.saturation_weight}")
    if args.save_preprocessed:
        print(f"preprocessed={args.output / 'preprocessed_fire.npz'}")
    print(f"stats={stats_path}")
    print(f"previews={args.output / 'prediction_*.png'}")
    print(f"comparison={args.output / 'fit_comparison.mp4'}")
    print(f"profiles={args.output / 'profiles.npz'}")


if __name__ == "__main__":
    main()
