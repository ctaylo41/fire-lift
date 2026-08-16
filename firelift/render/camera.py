from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass
class RayBundle:
    """A dense image-shaped collection of rays.

    Shapes:
        origins:    [H, W, 3]
        directions: [H, W, 3]

    Directions should be unit length unless you have a deliberate reason
    otherwise.
    """

    origins: Tensor
    directions: Tensor


@dataclass
class OrthographicCamera:
    """Orthographic camera with camera-to-world pose.

    Convention:
        camera-local +x = image right
        camera-local +y = image up
        camera-local +z = forward

    Attributes:
        R_wc: [3, 3] rotation mapping camera-frame vectors to world frame.
        t_wc: [3] camera origin in world coordinates.
        ortho_width: world-space width covered by the image plane.
        near: ray parameter at first sample.
        far: ray parameter at final sample.
    """

    R_wc: Tensor
    t_wc: Tensor
    ortho_width: float = 2.0
    near: float = 0.0
    far: float = 4.0

    def generate_rays(self, height: int, width: int) -> RayBundle:
        """Generate one orthographic ray per image pixel.

        TODO:
        1. Build image-plane coordinates centred around zero.
        2. Respect pixel aspect ratio when converting ortho_width to height.
        3. Form camera-space origins `(x_img, y_img, 0)`.
        4. Transform origins into world space with `R_wc` and `t_wc`.
        5. Transform camera forward `(0, 0, 1)` into world space.
        6. Broadcast the forward vector to every pixel.
        """
        aspect_ratio = width / height
        ortho_height = self.ortho_width / aspect_ratio
        
        x_coords = torch.linspace(-self.ortho_width / 2, self.ortho_width / 2, width)
        y_coords = torch.linspace(-ortho_height / 2, ortho_height / 2, height)
        
        x_img, y_img = torch.meshgrid(x_coords, y_coords, indexing='ij')
        
        z_img = torch.zeros_like(x_img)  
        camera_origins = torch.stack([x_img, y_img, z_img], dim=-1)
        
        world_origins = camera_origins @ self.R_wc.T + self.t_wc
        
        camera_forward = torch.tensor([0.0, 0.0, 1.0])
        world_forward = camera_forward @ self.R_wc.T  # [3]
        world_directions = torch.broadcast_to(world_forward, (height, width, 3))
        
        return RayBundle(origins=world_origins, directions=world_directions)
        
    @classmethod
    def look_at(
        cls,
        eye: Tensor,
        target: Tensor,
        up: Tensor,
        *,
        ortho_width: float = 2.0,
        near: float = 0.0,
        far: float = 4.0,
    ) -> "OrthographicCamera":
        """Construct a camera from an eye point, target, and approximate up.

        TODO:
        Build an orthonormal camera frame such that camera +z points from
        `eye` toward `target`. Be explicit about cross-product order and test
        handedness with a simple scene.
        """
        # Build orthonormal camera frame
        # f: forward direction from eye toward target
        f = torch.nn.functional.normalize(target - eye, dim=-1)
        
        # r: rightward direction (perpendicular to both up and f)
        r = torch.nn.functional.normalize(torch.cross(up, f, dim=-1), dim=-1)
        
        # u: upward direction (perpendicular to both f and r)
        u = torch.cross(f, r, dim=-1)
        
        # R_wc: rotation matrix with basis vectors as rows
        R_wc = torch.stack([r, u, f], dim=0)  # [3, 3]
        
        # t_wc: camera position in world coordinates
        t_wc = eye
        
        return cls(R_wc=R_wc, t_wc=t_wc, ortho_width=ortho_width, near=near, far=far)
        
        
        
        
        
