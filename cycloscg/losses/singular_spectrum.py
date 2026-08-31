from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def normalized_singular_spectrum(cycles: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if cycles.ndim != 3:
        raise ValueError("Expected cycles with shape [B, K, L]")
    centered = cycles - cycles.mean(dim=-1, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    return singular_values / singular_values.sum(dim=-1, keepdim=True).clamp_min(eps)


class SingularSpectrumLoss(nn.Module):
    """Match normalized singular spectra without imposing rank one."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(
            normalized_singular_spectrum(estimate, self.eps),
            normalized_singular_spectrum(target, self.eps),
        )

