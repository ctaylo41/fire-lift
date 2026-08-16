import torch
import numpy as np

from firelift.render.raymarch import render_emission
from firelift.volume.dense import DenseVolume
from firelift.render.camera import OrthographicCamera


def test_render_gradient_matches_finite_difference() -> None:
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        resolution = (4, 4, 4)
        bounds = (-1.0, 1.0)
        vol = DenseVolume(resolution=resolution, init_value=1.0, bounds=bounds)

        R_wc = torch.eye(3)
        t_wc = torch.tensor([0.0, 0.0, -2.5])
        camera = OrthographicCamera(R_wc=R_wc, t_wc=t_wc, ortho_width=2.2, near=1.5, far=3.5)
        height, width = 8, 8

        # Autograd gradients
        vol.theta.requires_grad_(True)
        loss = render_emission(vol, camera, height, width, n_samples=96).mean()
        loss.backward()
        autograd_grads = vol.theta.grad.clone()

        # Finite difference gradients
        epsilon = 1e-6
        numerical_grads = torch.zeros_like(autograd_grads)

        theta_original = vol.theta.data.clone()

        with torch.no_grad():
            for idx in np.ndindex(tuple(vol.theta.shape)):
                vol.theta.data = theta_original.clone()
                vol.theta.data[idx] += epsilon
                loss_pos = render_emission(vol, camera, height, width, n_samples=96).mean()

                vol.theta.data = theta_original.clone()
                vol.theta.data[idx] -= epsilon
                loss_neg = render_emission(vol, camera, height, width, n_samples=96).mean()

                numerical_grads[idx] = (loss_pos - loss_neg) / (2 * epsilon)

        vol.theta.data = theta_original

        assert torch.allclose(autograd_grads, numerical_grads, atol=1e-6)
    finally:
        torch.set_default_dtype(original_dtype)

    
    
    
        
    
    
    
    
