from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float


class ToyLoss(nn.Module):
    """
    Apply 1/z-weighted L1 loss only to the first points from the camera
    """
    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(
            self,
            input_dict: dict[str, torch.Tensor],
            output_dict: dict[str, torch.Tensor],
        ) -> torch.Tensor:
        input_points = input_dict["points_rendered"]  # (b, h, w, layers, 3)
        target_points = output_dict["points"]  # (b, h, w, layers, 3)
        input_masks = input_dict.get("masks_rendered", None)  # (b, h, w)

        batch, height, width, layers, channel = input_points.shape
        assert channel == 3
        assert input_points.shape == target_points.shape, f"{input_points.shape=}, {target_points.shape=}"
        assert input_masks is None or (batch, height, width) == input_masks.shape, f"{input_masks.shape=}"

        # only consider the most forefront points
        input_first = input_points[:, :, :, 0, :]
        target_first = target_points[:, :, :, 0, :]

        if input_masks is not None:
            diff = torch.abs(input_first[input_masks, :] - target_first[input_masks, :])
            loss_per_pixel = diff / input_first[input_masks][:, 2:3]
        else:
            diff = torch.abs(input_first - target_first)
            loss_per_pixel = diff / input_first[:, :, :, 2:3]

        return self.weight * torch.mean(loss_per_pixel)
