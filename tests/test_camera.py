import torch
import torch.nn.functional as F

from firelift.render.camera import OrthographicCamera
from firelift.volume.dense import DenseVolume


def test_centre_ray_points_at_target() -> None:
    """The center ray should point toward the camera forward direction."""
    # Create identity camera (no rotation, at origin)
    R_wc = torch.eye(3)
    t_wc = torch.zeros(3)
    
    camera = OrthographicCamera(R_wc=R_wc, t_wc=t_wc, ortho_width=2.0, near=0.0, far=4.0)
    
    height, width = 32, 32
    rays = camera.generate_rays(height, width)
    
    center_idx = (height // 2, width // 2)
    center_ray_direction = rays.directions[center_idx]
    
    expected_direction = torch.tensor([0.0, 0.0, 1.0])
    
    assert torch.allclose(center_ray_direction, expected_direction, atol=1e-6)


def test_orthographic_directions_are_parallel() -> None:
    """Every orthographic ray should have the same direction."""
    R_wc = torch.eye(3)
    t_wc = torch.zeros(3)
    
    camera = OrthographicCamera(R_wc=R_wc, t_wc=t_wc, ortho_width=2.0, near=0.0, far=4.0)
    
    height, width = 32, 32
    rays = camera.generate_rays(height, width)
    
    directions = rays.directions  # [H, W, 3]
    first_direction = directions[0, 0]
    
    # Check that all rays have the same direction
    assert torch.allclose(directions, first_direction.unsqueeze(0).unsqueeze(0), atol=1e-6)


def test_non_square_camera_has_hw_layout() -> None:
    """A non-square image should preserve height/width layout in the ray grid."""
    camera = OrthographicCamera(R_wc=torch.eye(3), t_wc=torch.zeros(3), ortho_width=2.0)
    height, width = 5, 9

    rays = camera.generate_rays(height, width)

    assert rays.origins.shape == (height, width, 3)
    assert rays.directions.shape == (height, width, 3)
    assert torch.allclose(rays.origins[:, :, 2], torch.zeros((height, width)))
    assert rays.origins[..., 0].shape == (height, width)
    assert rays.origins[..., 1].shape == (height, width)


def test_look_at_forward_matches_target_direction() -> None:
    """The center ray should point in the same direction as the look-at target."""
    eye = torch.tensor([0.0, 0.0, 0.0])
    target = torch.tensor([2.0, 3.0, 4.0])
    up = torch.tensor([0.0, 1.0, 0.0])

    camera = OrthographicCamera.look_at(eye, target, up)
    rays = camera.generate_rays(16, 16)

    expected = F.normalize(target - eye, dim=-1)
    center = rays.directions[8, 8]

    assert torch.allclose(center, expected, atol=1e-6)


def test_dense_materialize_matches_emission_grid() -> None:
    """Materializing a dense volume at the lattice points should match the emission grid."""
    volume = DenseVolume((4, 5, 6), init_value=0.5)

    expected = volume.emission_grid()
    materialized = volume.materialize((4, 5, 6), (-1.0, 1.0))

    assert materialized.shape == expected.shape
    assert torch.allclose(materialized, expected, atol=1e-6, rtol=1e-6)
