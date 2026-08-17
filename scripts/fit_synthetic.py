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
from firelift.synth.generate import make_plume_volume
from firelift.volume.dense import DenseVolume
from firelift.volume.FixedDenseVolume import FixedDenseVolume

def _save_image(path: Path, image: torch.Tensor) -> None:
    arr = image.detach().cpu().numpy()
    if arr.ndim == 2:
        arr = np.clip(arr, 0.0, None)
    elif arr.ndim == 3:
        arr = np.clip(arr, 0.0, None)
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

    gt_tensor = make_plume_volume(resolution, seed=seed)
    gt_volume = FixedDenseVolume(gt_tensor)

    gt_camera = OrthographicCamera(R_wc=torch.eye(3), t_wc=torch.tensor([0.0, 0.0, -2.5]), near=1.5, far=3.5)

    reconstruction_volume = DenseVolume(resolution)
    target_image = render_emission(gt_volume, gt_camera, height, width, n_samples=64)

    observation = [Observation(target_image, gt_camera)]
    weights = LossWeights(image=1.0, sparsity=1.0, tv=1.0)
    config = FitConfig(steps=100, lr=1e-2, n_samples_per_ray=64, log_every=5)

    history = fit_volume(reconstruction_volume, observation, weights=weights, config=config)
    predicted_image = render_emission(reconstruction_volume, gt_camera, height, width, n_samples=64)
    predicted_volume = reconstruction_volume.materialize(resolution, bounds=(-1.0, 1.0))

    img_err = image_l1(predicted_image, target_image)
    vol_err = normalized_volume_l1(predicted_volume, gt_volume)

    print(f"image_l1={img_err.item():.6f}")
    print(f"normalized_volume_l1={vol_err.item():.6f}")
    print(f"loss_history_steps={len(history['total'])}")

    _save_image(out_dir / "target.png", target_image)
    _save_image(out_dir / "predicted.png", predicted_image)
    _save_loss_curve(out_dir / "loss_curve.png", history)
    _save_volume_slices(out_dir, predicted_volume)
    print(f"Saved synthetic-fit artifacts to {out_dir}")


if __name__ == "__main__":
    main()
