import torch

from firelift.render.camera import OrthographicCamera


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
