# fire-lift — Part 1 skeleton

Goal: recover a plausible 3D **emission volume** from one or more images of an isolated flame using a differentiable emission-only renderer and strong structural priors.

## Coordinate convention

- World points are always `(x, y, z)`.
- World `+z` is vertical / flame height.
- Axisymmetric volumes revolve around world `z`, so `r = sqrt(x^2 + y^2)`.
- Camera-local axes are `+x = right`, `+y = up`, `+z = forward`.
- Dense tensors use PyTorch volume memory order `[D, H, W] = [z, y, x]`.
- `torch.nn.functional.grid_sample` still expects sampling coordinates in `(x, y, z)` order.
- Default volume bounds are `[-1, 1]^3` unless you deliberately change them.

## Suggested implementation order

1. `firelift/render/camera.py`
2. `firelift/volume/dense.py`
3. `firelift/render/raymarch.py`
4. analytic renderer tests
5. `firelift/synth/generate.py`
6. single-view dense reconstruction
7. multi-view experiment
8. TV + sparsity
9. `AxisymmetricVolume`
10. real-video preprocessing / fitting

All core numerical methods are intentionally left incomplete.
