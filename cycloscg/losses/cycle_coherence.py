from __future__ import annotations

import torch
from torch import nn


def pairwise_cycle_correlation(cycles: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return per-example ``[K, K]`` correlation matrices for ``[B, K, L]``."""
    if cycles.ndim != 3:
        raise ValueError("Expected cycles with shape [B, K, L]")
    centered = cycles - cycles.mean(dim=-1, keepdim=True)
    normalized = centered / torch.linalg.vector_norm(centered, dim=-1, keepdim=True).clamp_min(eps)
    return normalized @ normalized.transpose(-1, -2)


class CrossCycleCoherenceLoss(nn.Module):
    """Match the clean target's beat-correlation structure.

    This does not maximize all correlations: doing so would erase legitimate
    beat-to-beat physiological variation. It matches the target correlation
    matrix instead.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        difference = pairwise_cycle_correlation(estimate, self.eps) - pairwise_cycle_correlation(
            target, self.eps
        )
        cycles = estimate.shape[1]
        return torch.linalg.matrix_norm(difference, ord="fro", dim=(-2, -1)).mean() / cycles

