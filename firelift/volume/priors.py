from __future__ import annotations

from torch import Tensor


def sparsity_l1(field: Tensor) -> Tensor:
    """Mean absolute emission magnitude.

    For a non-negative field this is equivalent to mean emission.
    """
    raise NotImplementedError


def total_variation(field: Tensor) -> Tensor:
    """Anisotropic total variation for a 2D or 3D scalar field.

    TODO:
        Support `[H,R]` and `[D,H,W]` by summing mean absolute neighbour
        differences along every spatial dimension.
    """
    raise NotImplementedError


def temporal_l1(current: Tensor, previous: Tensor) -> Tensor:
    """Simple temporal consistency prior between consecutive fields.

    Keep this weak: real fire changes. This term is for suppressing arbitrary
    fitting flicker, not freezing the flame.
    """
    raise NotImplementedError
