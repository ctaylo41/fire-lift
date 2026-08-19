from firelift.volume.dense import DenseVolume
from firelift.render.camera import OrthographicCamera
from firelift.render.raymarch import render_emission
from firelift.fit.reconstruct import Observation 
from firelift.fit.reconstruct import compute_loss
from firelift.fit.reconstruct import LossWeights


import torch

def test_image_only_compute_loss_returns_valid_tuple():
    resolution = (32, 32, 32)
    init_value = 1.0
    bounds = (-1.0, 1.0)
    vol = DenseVolume(resolution=resolution, init_value=init_value, bounds=bounds)
    
    R_wc = torch.eye(3)
    t_wc = torch.tensor([0.0, 0.0, -2.5])
    camera = OrthographicCamera(R_wc=R_wc, t_wc=t_wc, ortho_width=2.2, near=1.5, far=3.5)
    height, width = 32, 32
    
    image = render_emission(vol, camera, height, width,n_samples=96)
    obs = Observation(image, camera)
    
    config = LossWeights(1.0, 0, 0)
    
    loss = compute_loss(vol, [obs] ,weights=config, n_samples_per_ray=96)
    
    assert loss[0].ndim == 0
    
    assert "image" in loss[1]
    assert "sparsity" in loss[1]
    assert "variation" in loss[1]
    assert len(loss[1].keys())==3

    for key in loss[1].keys():
        assert loss[1][key].ndim == 0


def test_prior_enabled_compute_loss_uses_regularization_field():
    resolution = (32, 32, 32)
    init_value = 1.0
    bounds = (-1.0, 1.0)
    vol = DenseVolume(resolution=resolution, init_value=init_value, bounds=bounds)
    
    # Add variation to the volume so TV is non-zero
    d, h, w = resolution
    z = torch.linspace(-1.0, 1.0, d).view(1, 1, d, 1, 1)
    vol.theta.data = (2.0 * z).expand(1, 1, d, h, w).contiguous()
    
    R_wc = torch.eye(3)
    t_wc = torch.tensor([0.0, 0.0, -2.5])
    camera = OrthographicCamera(R_wc=R_wc, t_wc=t_wc, ortho_width=2.2, near=1.5, far=3.5)
    height, width = 32, 32
    
    image = render_emission(vol, camera, height, width,n_samples=96)
    obs = Observation(image, camera)
    
    config_one = LossWeights(1.0, 0, 0)
    config_two = LossWeights(1.0,1.0,1.0)
    loss_one = compute_loss(vol, [obs], weights=config_one, n_samples_per_ray=96)
    loss_two = compute_loss(vol, [obs],weights=config_two, n_samples_per_ray=96)
    
    assert loss_two[1]["sparsity"] > 0
    assert loss_two[1]["variation"] > 0
    assert loss_one[1]["sparsity"] == 0
    assert loss_one[1]["variation"] == 0

def test_two_identical_observations_have_same_mean_image_loss_as_one():
    resolution = (32, 32, 32)
    init_value = 1.0
    bounds = (-1.0, 1.0)
    vol = DenseVolume(resolution=resolution, init_value=init_value, bounds=bounds)

    R_wc = torch.eye(3)
    t_wc = torch.tensor([0.0, 0.0, -2.5])
    camera = OrthographicCamera(R_wc=R_wc, t_wc=t_wc, ortho_width=2.2, near=1.5, far=3.5)
    height, width = 32, 32
    
    target_image = torch.ones(height, width) * 0.5
    
    obs_1 = Observation(target_image, camera)
    obs_2 = Observation(target_image, camera)
    config = LossWeights(1.0, 0, 0)
    
    loss_one = compute_loss(vol, [obs_1], weights=config, n_samples_per_ray=96)
    loss_two = compute_loss(vol, [obs_1, obs_2], weights=config, n_samples_per_ray=96)
    
    assert torch.allclose(loss_one[1]["image"], loss_two[1]["image"])
    