"""Run a multi-view ablation over an asymmetric diagnostic GT and summarize results."""
from __future__ import annotations

import csv
from pathlib import Path

import imageio.v3 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from firelift.eval.metrics import image_l1, normalized_volume_l1
from firelift.fit.reconstruct import FitConfig, LossWeights, Observation, fit_volume
from firelift.render.camera import OrthographicCamera
from firelift.render.raymarch import render_emission
from firelift.synth.generate import make_asymmetric_diagnostic_volume
from firelift.volume.dense import DenseVolume
from firelift.volume.FixedDenseVolume import FixedDenseVolume


def _save_image(path: Path, image: torch.Tensor) -> None:
    arr = image.detach().cpu().numpy()
    arr = np.clip(arr, 0.0, None)
    arr = (arr * 255).astype(np.uint8)
    imageio.imwrite(path, arr)


def make_camera_from_angle(
    angle_deg: float,
    *,
    radius: float = 2.5,
    target: torch.Tensor | None = None,
    up: torch.Tensor | None = None,
    ortho_width: float = 2.2,
    near: float = 1.5,
    far: float = 3.5,
) -> OrthographicCamera:
    if target is None:
        target = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    if up is None:
        up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

    theta = np.deg2rad(angle_deg)
    eye = torch.tensor(
        [radius * np.sin(theta), -radius * np.cos(theta), 0.0],
        dtype=torch.float32,
    )
    return OrthographicCamera.look_at(eye, target, up, ortho_width=ortho_width, near=near, far=far)


def render_camera_set(
    volume: FixedDenseVolume | DenseVolume,
    cameras: list[OrthographicCamera],
    *,
    height: int,
    width: int,
    n_samples: int,
) -> list[torch.Tensor]:
    return [
        render_emission(volume, camera, height, width, n_samples=n_samples)
        for camera in cameras
    ]


def save_volume_slices(path: Path, volume: torch.Tensor) -> None:
    path.mkdir(parents=True, exist_ok=True)
    volume_np = volume.detach().cpu().numpy()
    z_indices = np.linspace(0, volume_np.shape[0] - 1, min(4, volume_np.shape[0]), dtype=int)
    for idx, z in enumerate(z_indices):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(volume_np[z], cmap="magma")
        ax.set_title(f"slice z={z}")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path / f"volume_slice_{idx}.png")
        plt.close(fig)


def save_image_pair(path: Path, label: str, gt_image: torch.Tensor, recon_image: torch.Tensor) -> None:
    _save_image(path / f"{label}_gt.png", gt_image)
    _save_image(path / f"{label}_recon.png", recon_image)


def main() -> None:
    seed = 42
    resolution = (32, 32, 32)
    height, width = (32, 32)
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "synthetic_fit" / "view_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_tensor = make_asymmetric_diagnostic_volume(resolution, seed=seed)
    gt_volume = FixedDenseVolume(gt_tensor)

    target = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

    observation_count = 16
    observation_angles = np.linspace(0.0, 360.0, observation_count, endpoint=False)
    novel_step = 360.0 / observation_count / 2.0
    novel_angles = (observation_angles + novel_step) % 360.0

    observation_cameras = [
        make_camera_from_angle(angle, target=target, up=up) for angle in observation_angles
    ]
    novel_cameras = [
        make_camera_from_angle(angle, target=target, up=up) for angle in novel_angles
    ]

    observation_images = render_camera_set(
        gt_volume,
        observation_cameras,
        height=height,
        width=width,
        n_samples=64,
    )
    novel_images = render_camera_set(
        gt_volume,
        novel_cameras,
        height=height,
        width=width,
        n_samples=64,
    )

    summary_rows: list[dict[str, float]] = []

    for n_views in [1, 2, 4, 8, 16]:
        run_dir = out_dir / f"n_{n_views}"
        run_dir.mkdir(parents=True, exist_ok=True)
        recon_dir = run_dir / "recon"
        gt_dir = run_dir / "gt"
        recon_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

        selected_cameras = observation_cameras[:n_views]
        selected_images = observation_images[:n_views]
        observations = [
            Observation(image, camera)
            for image, camera in zip(selected_images, selected_cameras)
        ]

        recon_volume = DenseVolume(resolution)
        weights = LossWeights(image=1.0, sparsity=0.0, tv=0.0)
        config = FitConfig(steps=1000, lr=1e-2, n_samples_per_ray=64, log_every=50)

        history = fit_volume(recon_volume, observations, weights=weights, config=config)
        pred_volume = recon_volume.materialize(resolution, bounds=(-1.0, 1.0))
        gt_volume_tensor = gt_volume.materialize(resolution, bounds=(-1.0, 1.0))

        observed_err = 0.0
        for camera, gt_image in zip(selected_cameras, selected_images):
            pred_image = render_emission(recon_volume, camera, height, width, n_samples=64)
            observed_err += image_l1(pred_image, gt_image).item()
        observed_err /= len(selected_cameras)

        novel_err = 0.0
        for camera, gt_image in zip(novel_cameras, novel_images):
            pred_image = render_emission(recon_volume, camera, height, width, n_samples=64)
            novel_err += image_l1(pred_image, gt_image).item()
        novel_err /= len(novel_cameras)

        volume_err = normalized_volume_l1(pred_volume, gt_volume_tensor).item()
        summary_rows.append(
            {
                "n_views": float(n_views),
                "observed_l1": float(observed_err),
                "novel_l1": float(novel_err),
                "volume_l1": float(volume_err),
            }
        )

        first_obs_gt = selected_images[0]
        first_obs_pred = render_emission(recon_volume, selected_cameras[0], height, width, n_samples=64)
        save_image_pair(run_dir, "observed_0", first_obs_gt, first_obs_pred)

        first_novel_gt = novel_images[0]
        first_novel_pred = render_emission(recon_volume, novel_cameras[0], height, width, n_samples=64)
        save_image_pair(run_dir, "novel_0", first_novel_gt, first_novel_pred)

        save_volume_slices(recon_dir, pred_volume)
        save_volume_slices(gt_dir, gt_volume_tensor)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history.get("total", []), label="total")
        ax.plot(history.get("image", []), label="image")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "loss_curve.png")
        plt.close(fig)

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n_views", "observed_l1", "novel_l1", "volume_l1"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print("n_views\tobserved_l1\tnovel_l1\tvolume_l1")
    for row in summary_rows:
        print(f"{int(row['n_views'])}\t{row['observed_l1']:.6f}\t{row['novel_l1']:.6f}\t{row['volume_l1']:.6f}")

    ns = [int(row["n_views"]) for row in summary_rows]
    observed = [row["observed_l1"] for row in summary_rows]
    novel = [row["novel_l1"] for row in summary_rows]
    volume = [row["volume_l1"] for row in summary_rows]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(ns, observed, "o-")
    axes[0].set_title("Observed-view L1")
    axes[0].set_xlabel("N views")
    axes[0].set_ylabel("L1")

    axes[1].plot(ns, novel, "o-")
    axes[1].set_title("Novel-view L1")
    axes[1].set_xlabel("N views")
    axes[1].set_ylabel("L1")

    axes[2].plot(ns, volume, "o-")
    axes[2].set_title("Volume L1")
    axes[2].set_xlabel("N views")
    axes[2].set_ylabel("L1")

    fig.tight_layout()
    fig.savefig(out_dir / "view_ablation_summary.png")
    plt.close(fig)

    print(f"Saved summary at {summary_path}")


if __name__ == "__main__":
    main()
