"""Export compact fitted profiles as one OpenVDB FloatGrid per frame."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from firelift.volume.axisymmetric import AxisymmetricVolume, FourierVolume


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", type=Path, help="profiles.npz from fit_video.py")
    parser.add_argument("--output", type=Path, default=Path("outputs/vdb"))
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--threshold-fraction", type=float, default=0.005)
    parser.add_argument("--voxel-size", type=float, default=1.0)
    return parser.parse_args()


def load_openvdb():
    try:
        import pyopenvdb as vdb
    except ImportError as exc:
        raise RuntimeError(
            "OpenVDB Python bindings are required. Install `py-openvdb` from conda-forge "
            "in the environment running this script."
        ) from exc
    return vdb


def build_volume(data: dict[str, np.ndarray], frame_index: int):
    representation = str(data["representation"])
    kwargs = {
        "profile_resolution": (int(data["profile_height"]), int(data["profile_radius"])),
        "max_radius": float(data["max_radius"]),
        "z_bounds": (float(data["z_min"]), float(data["z_max"])),
    }
    volume = FourierVolume(**kwargs) if representation == "fourier1" else AxisymmetricVolume(**kwargs)
    theta = torch.from_numpy(data["theta"][frame_index]).to(volume.theta)
    volume.theta.data.copy_(theta)
    return volume


def main() -> None:
    args = parse_args()
    if args.resolution <= 0:
        raise ValueError("resolution must be positive")
    if not 0.0 <= args.threshold_fraction:
        raise ValueError("threshold-fraction must be non-negative")

    vdb = load_openvdb()
    data = dict(np.load(args.profiles, allow_pickle=False))
    theta = data["theta"]
    frame_count = theta.shape[0]
    resolution = (args.resolution, args.resolution, args.resolution)
    volumes = [build_volume(data, index).materialize(resolution).detach().cpu().numpy() for index in range(frame_count)]
    sequence_max = max(float(volume.max()) for volume in volumes)
    threshold = args.threshold_fraction * sequence_max
    args.output.mkdir(parents=True, exist_ok=True)

    for index, volume in enumerate(volumes, start=1):
        vdb_array = np.transpose(volume, (2, 1, 0)).astype(np.float32, copy=True)
        vdb_array[vdb_array < threshold] = 0.0
        grid = vdb.FloatGrid(background=0.0)
        grid.copyFromArray(vdb_array)
        grid.name = "emission"
        if hasattr(vdb, "createLinearTransform"):
            grid.transform = vdb.createLinearTransform(voxelSize=args.voxel_size)
        output_path = args.output / f"fire_{index:04d}.vdb"
        vdb.write(str(output_path), grids=[grid])
        print(f"wrote {output_path} active_max={float(vdb_array.max()):.6f}")


if __name__ == "__main__":
    main()