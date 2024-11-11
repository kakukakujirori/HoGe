from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float


class L2Loss(nn.Module):
    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None):
        assert inputs.shape == targets.shape

        if mask is None:
            loss = self.mse_loss(inputs, targets)
        else:
            loss = self.mse_loss(inputs[mask], targets[mask])

        return self.weight * loss