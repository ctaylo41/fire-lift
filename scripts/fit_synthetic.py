"""Run the core synthetic inverse-rendering experiments.

Suggested CLI/config dimensions:
- representation: dense | axisymmetric
- views: 1 | 2 | 4 | 8 | 16
- lambda_tv
- lambda_sparse
- seed

Outputs to save:
- fitted volume/profile parameters
- loss history
- rendered reconstructions
- image error
- normalized GT volume error
"""

from __future__ import annotations

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
    if arr.ndim == 2:
        arr = np.clip(arr, 0.0, None)
    elif arr.ndim == 3:
        arr = np.clip(arr, 0.0, None)
    arr = (arr * 255).astype(np.uint8)
    imageio.imwrite(path, arr)


def _save_loss_curve(path: Path, history: dict[str, list[float]]) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(history.get("total", []), label="total")
    plt.plot(history.get("image", []), label="image")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Synthetic fit loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _save_volume_slices(path: Path, volume: torch.Tensor) -> None:
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


def main() -> None:
    seed = 42
    resolution = (32, 32, 32)
    height, width = (32, 32)
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "synthetic_fit"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_tensor = make_asymmetric_diagnostic_volume(resolution, seed=seed)
    gt_volume = FixedDenseVolume(gt_tensor)

    # Side view camera: looking toward origin with +z up
    eye = torch.tensor([0.0, -2.5, 0.0])
    target = torch.tensor([0.0, 0.0, 0.0])
    up = torch.tensor([0.0, 0.0, 1.0])
    gt_camera = OrthographicCamera.look_at(eye, target, up, ortho_width=2.2, near=1.5, far=3.5)

    reconstruction_volume = DenseVolume(resolution)
    target_image = render_emission(gt_volume, gt_camera, height, width, n_samples=64)

    observation = [Observation(target_image, gt_camera)]
    weights = LossWeights(image=1.0, sparsity=0.0, tv=0.0)
    config = FitConfig(steps=1000, lr=1e-2, n_samples_per_ray=64, log_every=5)

    history = fit_volume(reconstruction_volume, observation, weights=weights, config=config)
    predicted_image = render_emission(reconstruction_volume, gt_camera, height, width, n_samples=64)
    predicted_volume = reconstruction_volume.materialize(resolution, bounds=(-1.0, 1.0))
    gt_volume_tensor = gt_volume.materialize(resolution, bounds=(-1.0, 1.0))

    img_err = image_l1(predicted_image, target_image)
    vol_err = normalized_volume_l1(predicted_volume, gt_volume_tensor)



    print(f"image_l1={img_err.item():.6f}")
    print(f"normalized_volume_l1={vol_err.item():.6f}")
    print(f"loss_history_steps={len(history['total'])}")

    _save_image(out_dir / "target.png", target_image)
    _save_image(out_dir / "predicted.png", predicted_image)
    _save_loss_curve(out_dir / "loss_curve.png", history)
    _save_volume_slices(out_dir, predicted_volume)
    
    render_novel_views(gt_volume, reconstruction_volume, gt_camera, [0, 15, 30, 45, 60, 75, 90], height, width, out_dir, 64)
    
    print(f"Saved synthetic-fit artifacts to {out_dir}")

def render_novel_views(
    gt_volume: FixedDenseVolume,
    recon_volume: DenseVolume,
    base_camera: OrthographicCamera,
    angles: list[float],
    height: int,
    width: int,
    out_dir: Path,
    n_samples: int = 64
):
    """Render novel views by orbiting around the z-axis."""
    dtype = base_camera.R_wc.dtype
    device = base_camera.R_wc.device
    
    target = torch.tensor([0.0, 0.0, 0.0], dtype=dtype, device=device)
    up = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
    radius = 2.5
    
    for angle in angles:
        # Convert angle to radians and compute eye position on circle around z-axis
        angle_rad = angle * np.pi / 180
        eye_x = radius * torch.sin(torch.tensor(angle_rad, dtype=dtype, device=device))
        eye_y = -radius * torch.cos(torch.tensor(angle_rad, dtype=dtype, device=device))
        eye = torch.tensor([eye_x, eye_y, 0.0], dtype=dtype, device=device)
        
        # Create camera looking toward origin with consistent up direction
        novel_camera = OrthographicCamera.look_at(eye, target, up, ortho_width=2.2, near=1.5, far=3.5)
        
        image_ground_truth = render_emission(gt_volume, novel_camera, height, width, n_samples=n_samples)
        image_reconstructed = render_emission(recon_volume, novel_camera, height, width, n_samples=n_samples)
        
        _save_image(out_dir / f"image_gt_{angle}.png", image_ground_truth)
        _save_image(out_dir / f"images_recon_{angle}.png", image_reconstructed)
        
        

if __name__ == "__main__":
    main()
