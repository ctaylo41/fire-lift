import torch

from firelift.render.raymarch import render_emission
from firelift.volume.dense import DenseVolume
from firelift.render.camera import OrthographicCamera

def test_empty_volume_renders_zero() -> None:
    resolution = (32, 32, 32)
    init_value = -20.0
    bounds = (-1.0, 1.0)
    vol = DenseVolume(resolution=resolution, init_value=init_value, bounds=bounds)
    
    R_wc = torch.eye(3)
    t_wc = torch.zeros(3)
    camera = OrthographicCamera(R_wc=R_wc, t_wc=t_wc, ortho_width=2.0, near=0.0, far=4.0)
    height, width = 32, 32    
    image = render_emission(vol, camera, height, width, n_samples=96)
    
    assert torch.allclose(image, torch.tensor(0.0), atol=1e-5)



def test_constant_cube_matches_chord_integral() -> None:
    resolution = (32, 32, 32)
    init_value = 1.0
    bounds = (-1.0, 1.0)
    vol = DenseVolume(resolution=resolution, init_value=init_value, bounds=bounds)
    
    R_wc = torch.eye(3)
    t_wc = torch.tensor([0.0, 0.0, -2.5])
    camera = OrthographicCamera(R_wc=R_wc, t_wc=t_wc, ortho_width=2.2, near=1.5, far=3.5)

    height, width = 32, 32    
    image = render_emission(vol, camera, height, width, n_samples=96)
    
    chord_length = 2.0
    expected_value = chord_length * torch.nn.functional.softplus(torch.tensor(init_value))

    interior = image[3:29, 3:29]
    assert torch.allclose(interior, expected_value, atol=1e-5)

    


def test_more_samples_converge() -> None:
    resolution = (32, 32, 32)
    bounds = (-1.0, 1.0)
    vol = DenseVolume(resolution=resolution, init_value=1.0, bounds=bounds)

    d, h, w = resolution
    z = torch.linspace(-1.0, 1.0, d).view(1, 1, d, 1, 1)
    vol.theta.data = (3.0 * z).expand(1, 1, d, h, w).contiguous()

    R_wc = torch.eye(3)
    t_wc = torch.tensor([0.0, 0.0, -2.5])
    camera = OrthographicCamera(R_wc=R_wc, t_wc=t_wc, ortho_width=2.2, near=1.5, far=3.5)
    height, width = 32, 32

    reference = render_emission(vol, camera, height, width, n_samples=8192)[16, 16]

    errors = [
        (render_emission(vol, camera, height, width, n_samples=n)[16, 16] - reference).abs().item()
        for n in (8, 32, 128)
    ]

    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
