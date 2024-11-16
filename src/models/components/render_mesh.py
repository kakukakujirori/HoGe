import glob
import sys
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from kornia.morphology import opening
from pytorch3d.renderer import (
    BlendParams,
    MeshRasterizer,
    PerspectiveCameras,
    RasterizationSettings,
)
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.structures import Meshes


# https://github.com/facebookresearch/pytorch3d/issues/737
class VoidFillShader(torch.nn.Module):
    def __init__(
        self, void_color: tuple[float, float, float] = (0.0, 0.0, 0.0), void_alpha: float = 0.0
    ):
        super().__init__()
        assert len(void_color) == 3
        assert 0 <= min(void_color) and max(void_color) <= 1.0
        self.blend_params = BlendParams(
            background_color=(*void_color, void_alpha)
        )  # the last 0.0 means 'invalid'

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        texels = meshes.sample_textures(fragments)

        # Mask for the background.
        is_void = fragments.pix_to_face < 0  # (N, H, W, K)

        background_color_ = self.blend_params.background_color
        if isinstance(background_color_, torch.Tensor):
            void_color = background_color_.to(texels.device)
        else:
            void_color = texels.new_tensor(background_color_)  # pyre-fixme[16]

        texels[is_void] = void_color

        return texels


class RenderMesh(torch.nn.Module):
    def __init__(self, layer_num: int = 3, void_alpha: float = 0.0):
        super().__init__()
        self.layer_num = layer_num
        self.faces_per_pixel = layer_num * 2
        self.void_alpha = void_alpha

    def render(
        self,
        meshes: Meshes,
        cameras: PerspectiveCameras,
    ) -> tuple[Float[torch.Tensor, "b h w layer_num 4"], Float[torch.Tensor, "b h w layer_num"]]:

        image_size_list = [texture.shape[:2] for texture in meshes.textures.maps_list()]

        # Assuming you already have your mesh, cameras, etc. set up
        raster_settings = RasterizationSettings(
            image_size=image_size_list[0],
            blur_radius=1e-12,
            faces_per_pixel=self.faces_per_pixel,
            bin_size=None,
            cull_backfaces=False,  # This will ignore back-facing polygons
            perspective_correct=True,
        )

        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        )

        shader = VoidFillShader(void_alpha=self.void_alpha)

        # Get the rasterization outputs
        fragments = rasterizer(meshes)
        texels = shader(fragments, meshes)
        depths = fragments.zbuf

        return texels, depths

    def cull_redundant_layers(
        self,
        texels: Float[torch.Tensor, "b h w layers 4"],
        depths: Float[torch.Tensor, "b h w layers"],
        alpha_valid_area_thresh: float = 0.005,
        alpha_duplication_thresh: float = 0.99,
    ) -> tuple[Float[torch.Tensor, "b h w layers 4"], Float[torch.Tensor, "b h w layers"]]:

        batch_size, height, width, layer_num_ex, channels = texels.shape
        alpha = texels[..., 3:4]

        # noise reduction (opening)
        alpha = rearrange(alpha, "b h w layers c -> (b layers) c h w", c=1)
        alpha = opening(alpha, torch.ones(5, 5, device=alpha.device))
        alpha = rearrange(alpha, "(b layers) () h w -> b h w layers", b=batch_size)

        # detect void layers (NOTE: sky region should be treated as valid)
        alpha_valid_ratio = (alpha > 0.1).float().mean(dim=(1, 2))
        void_layer_indices = (alpha_valid_ratio < alpha_valid_area_thresh).nonzero(as_tuple=True)

        # detect duplication layers
        alpha_global_similarity = (
            (alpha[:, :, :, 1:] == alpha[:, :, :, :-1]).float().mean(dim=(1, 2))
        )
        duplication_layer_flag = alpha_global_similarity > alpha_duplication_thresh
        duplication_layer_indices = duplication_layer_flag.nonzero(as_tuple=True)
        duplication_layer_indices = (
            duplication_layer_indices[0],
            duplication_layer_indices[1] + 1,
        )

        # cull duplication layers
        valid_layer_mask = torch.ones(
            batch_size, layer_num_ex, dtype=torch.bool, device=texels.device
        )
        valid_layer_mask[void_layer_indices] = False
        valid_layer_mask[duplication_layer_indices] = False
        valid_layer_mask_batch, valid_layer_mask_layer = valid_layer_mask.nonzero(as_tuple=True)

        # discard layers that are more than self.layer_num
        target_layer_idx = self.section_arange(valid_layer_mask_batch)
        valid_layer_mask_limiter = target_layer_idx < self.layer_num
        valid_layer_mask_batch = valid_layer_mask_batch[valid_layer_mask_limiter]
        valid_layer_mask_layer = valid_layer_mask_layer[valid_layer_mask_limiter]
        target_layer_idx = target_layer_idx[valid_layer_mask_limiter]

        # cull texels and depths
        texels_culled = torch.zeros(
            batch_size, height, width, self.layer_num, 4, dtype=texels.dtype, device=texels.device
        )
        texels_culled[..., 3] = -1
        texels_culled[valid_layer_mask_batch, :, :, target_layer_idx, :] = texels[
            valid_layer_mask_batch, :, :, valid_layer_mask_layer, :
        ]

        depths_culled = torch.full(
            (batch_size, height, width, self.layer_num),
            fill_value=-1,
            dtype=depths.dtype,
            device=depths.device,
        )
        depths_culled[valid_layer_mask_batch, :, :, target_layer_idx] = depths[
            valid_layer_mask_batch, :, :, valid_layer_mask_layer
        ]

        return texels_culled, depths_culled

    @staticmethod
    def section_arange(x: torch.Tensor) -> torch.Tensor:
        "This function behaves like this: x=[0,0,0,1,1,2,4,4,4,5] -> y=[0,1,2,0,1,0,0,1,2,0]"
        assert len(x.shape) == 1 and torch.all(x[:-1] <= x[1:])

        # corner case
        if x.numel() == 0:
            return torch.arange(0, dtype=x.dtype, device=x.device)

        change_idx = torch.unique(x)
        change_val = torch.cat(
            [
                torch.tensor([0], dtype=x.dtype, device=x.device),
                (x[1:] != x[:-1]).nonzero().flatten() + 1,
            ]
        )
        table = torch.zeros(change_idx.max() + 1, dtype=x.dtype, device=x.device)
        table[change_idx] = change_val
        return torch.arange(len(x), dtype=x.dtype, device=x.device) - table[x]

    def forward(
        self,
        meshes: Meshes,
        cameras: PerspectiveCameras,
    ) -> tuple[Float[torch.Tensor, "b h w layers 4"], Float[torch.Tensor, "b h w layers"]]:
        texels, depths = self.render(meshes, cameras)
        texels, depths = self.cull_redundant_layers(texels, depths)
        depths[texels[..., 3] < 0.5] = -1  # mask invalid depth as well
        return texels, depths
