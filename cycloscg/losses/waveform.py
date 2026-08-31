from __future__ import annotations

import torch
from torch import nn


class WaveformLoss(nn.Module):
    def __init__(self, kind: str = "smooth_l1", beta: float = 1.0):
        super().__init__()
        kind = kind.lower()
        if kind == "smooth_l1":
            self.loss = nn.SmoothL1Loss(beta=beta)
        elif kind == "l1":
            self.loss = nn.L1Loss()
        else:
            raise ValueError("kind must be 'smooth_l1' or 'l1'")

    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(estimate, target)

