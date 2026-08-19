import torch

from firelift.render.camera import OrthographicCamera
from firelift.render.raymarch import render_emission
from firelift.synth.generate import sample_blob_params
from firelift.volume.axisymmetric import AxisymmetricVolume


def test_axisymmetric_samples_ignore_azimuth() -> None:
    """Points with equal `(r,z)` but different azimuth must have equal emission."""
    volume = AxisymmetricVolume(profile_resolution=(16, 8), max_radius=1.0, z_bounds=(-1.0, 1.0))
    volume.theta.data.fill_(1.0)

    points = torch.tensor(
        [
            [0.30, 0.00, 0.40],
            [0.00, 0.30, 0.40],
            [-0.30, 0.00, 0.40],
            [0.00, -0.30, 0.40],
        ],
        dtype=torch.float32,
    )

    samples = volume.sample(points)
    assert samples.shape == (4,)
    assert torch.allclose(samples[0], samples[1])
    assert torch.allclose(samples[0], samples[2])
    assert torch.allclose(samples[0], samples[3])


def test_axisymmetric_ninety_degree_rotation_same_render() -> None:
    """For equivalent cameras around z, a perfect axisymmetric field is invariant."""
    volume = AxisymmetricVolume(profile_resolution=(32, 16), max_radius=1.0, z_bounds=(-1.0, 1.0))
    volume.theta.data.copy_(torch.linspace(-2.0, 2.0, 32 * 16, dtype=torch.float32).reshape(1, 1, 32, 16))

    target = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

    camera_0 = OrthographicCamera.look_at(
        torch.tensor([0.0, -2.5, 0.0], dtype=torch.float32),
        target,
        up,
        ortho_width=2.2,
        near=1.5,
        far=3.5,
    )
    camera_90 = OrthographicCamera.look_at(
        torch.tensor([2.5, 0.0, 0.0], dtype=torch.float32),
        target,
        up,
        ortho_width=2.2,
        near=1.5,
        far=3.5,
    )

    render_0 = render_emission(volume, camera_0, 32, 32, n_samples=32)
    render_90 = render_emission(volume, camera_90, 32, 32, n_samples=32)

    assert render_0.shape == (32, 32)
    assert render_90.shape == (32, 32)
    assert torch.allclose(render_0, render_90, atol=1e-4, rtol=1e-4)


def test_axisymmetric_outside_volume_returns_zero() -> None:
    """Samples outside the compact cylindrical profile must be empty rather than boundary values."""
    volume = AxisymmetricVolume(profile_resolution=(16, 8), max_radius=1.0, z_bounds=(-1.0, 1.0))
    volume.theta.data.fill_(0.0)

    points = torch.tensor(
        [
            [1.5, 0.0, 0.0],
            [0.0, 0.0, 1.5],
            [0.0, 0.0, -1.5],
        ],
        dtype=torch.float32,
    )

    samples = volume.sample(points)
    assert samples.shape == (3,)
    assert torch.all(samples == 0.0)


def test_axisymmetric_blob_sampling_is_exactly_axisymmetric() -> None:
    """Axisymmetric sampling should keep all centers on the z-axis and equal x/y scales."""
    centers, scales, amplitudes = sample_blob_params(
        radial_spread=0.35,
        z_range=(-0.8, 0.7),
        scale_xy_range=(0.10, 0.28),
        scale_z_range=(0.15, 0.40),
        taper=0.55,
        num_samples=16,
        axisymmetric=True,
        axis_jitter=0.0,
        anisotropy=0.15,
        device="cpu",
        dtype=torch.float32,
    )

    assert centers.shape == (16, 3)
    assert torch.allclose(centers[:, 0], torch.zeros_like(centers[:, 0]))
    assert torch.allclose(centers[:, 1], torch.zeros_like(centers[:, 1]))
    assert torch.allclose(scales[:, 0], scales[:, 1])
    assert amplitudes.shape == (16,)
