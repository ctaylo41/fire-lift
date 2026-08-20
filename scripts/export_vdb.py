"""Export compact fitted profiles as one OpenVDB FloatGrid per frame."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from firelift.volume.axisymmetric import AxisymmetricVolume, BentAxisymmetricVolume, FourierVolume


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", type=Path, help="profiles.npz from fit_video.py")
    parser.add_argument("--output", type=Path, default=Path("outputs/vdb"))
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--threshold-fraction", type=float, default=0.0)
    parser.add_argument("--frame", type=int, default=None, help="One-based frame number to export")
    parser.add_argument("--voxel-size", type=float, default=1.0)
    parser.add_argument(
        "--center",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Center the voxel volume at world origin with z remaining vertical.",
    )
    return parser.parse_args()


def load_openvdb():
    try:
        import openvdb as vdb
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
    if representation == "fourier1":
        volume = FourierVolume(**kwargs)
    elif representation == "bent_axisymmetric":
        centerline_points = int(data.get("centerline_points", data["theta"].shape[-1]))
        volume = BentAxisymmetricVolume(**kwargs, centerline_points=centerline_points)
    else:
        volume = AxisymmetricVolume(**kwargs)
    theta = torch.from_numpy(data["theta"][frame_index]).to(volume.theta)
    volume.theta.data.copy_(theta)
    if representation == "bent_axisymmetric" and "centerline" in data:
        centerline = torch.from_numpy(data["centerline"][frame_index]).to(volume.centerline)
        volume.centerline.data.copy_(centerline)
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
    if args.frame is not None and not 1 <= args.frame <= frame_count:
        raise ValueError(f"frame must be between 1 and {frame_count}")
    frame_indices = [args.frame - 1] if args.frame is not None else list(range(frame_count))
    volumes = [
        build_volume(data, index).materialize(resolution).detach().cpu().numpy()
        for index in frame_indices
    ]
    sequence_max = max(float(volume.max()) for volume in volumes)
    threshold = args.threshold_fraction * sequence_max
    print(f"frames={len(volumes)} resolution={args.resolution} threshold={threshold:.9g}")
    args.output.mkdir(parents=True, exist_ok=True)

    for source_index, volume in zip(frame_indices, volumes):
        index = source_index + 1
        # This binding maps NumPy axes 0, 1, 2 to VDB i, j, k. Convert the
        # internal [z,y,x] layout to [x,y,z] so flame height lands on VDB k/z.
        vdb_array = np.transpose(volume, (2, 1, 0)).astype(np.float32, copy=True)
        print(
            f"frame={index} before_threshold min={vdb_array.min():.9g} "
            f"max={vdb_array.max():.9g} nonzero={np.count_nonzero(vdb_array)}"
        )
        vdb_array[vdb_array < threshold] = 0.0
        print(
            f"frame={index} after_threshold min={vdb_array.min():.9g} "
            f"max={vdb_array.max():.9g} nonzero={np.count_nonzero(vdb_array)}"
        )
        grid = vdb.FloatGrid(background=0.0)
        grid.name = "emission"
        grid.gridClass = vdb.GridClass.FOG_VOLUME
        grid.copyFromArray(vdb_array)
        if hasattr(vdb, "createLinearTransform"):
            transform = vdb.createLinearTransform(voxelSize=args.voxel_size)
            if args.center:
                center = 0.5 * (args.resolution - 1) * args.voxel_size
                transform.postTranslate((-center, -center, -center))
            grid.transform = transform
        output_path = args.output / f"fire_{index:04d}.vdb"
        vdb.write(str(output_path), grids=[grid])
        print(f"active_voxels={grid.activeVoxelCount()}")
        print(f"bbox={grid.evalActiveVoxelBoundingBox()}")
        print(f"transform={grid.transform}")
        print(
            f"index_origin_world={grid.transform.indexToWorld((0, 0, 0))} "
            f"index_center_world={grid.transform.indexToWorld((args.resolution - 1,) * 3)}"
        )
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()