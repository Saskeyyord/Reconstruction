from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def normalized_cycles(cycles: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    centered = cycles - cycles.mean(dim=-1, keepdim=True)
    rms = torch.sqrt(torch.mean(centered.square(), dim=-1, keepdim=True) + eps)
    return centered / rms


def phase_gram(cycles: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Cardiac-phase-dependent second-order statistics ``[B, L, L]``.

    Averaging the outer products over cycles estimates how normalized cardiac
    phases covary across consecutive beats. Matching this statistic is the
    central differentiable cyclostationarity constraint in phase coordinates.
    """
    values = normalized_cycles(cycles, eps)
    cycle_count = values.shape[1]
    return values.transpose(-1, -2) @ values / max(cycle_count, 1)


class PhaseCovarianceLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        estimate_gram = phase_gram(estimate, self.eps)
        target_gram = phase_gram(target, self.eps)
        numerator = torch.linalg.matrix_norm(
            estimate_gram - target_gram, ord="fro", dim=(-2, -1)
        )
        denominator = torch.linalg.matrix_norm(target_gram, ord="fro", dim=(-2, -1)).clamp_min(
            self.eps
        )
        return (numerator / denominator).mean()


class CyclicStatisticsLoss(nn.Module):
    """Experimental phase-lagged correlation matching.

    Each normalized cycle is treated in cardiac-phase coordinates; the loss
    matches phase-dependent lag products averaged across cycles for selected
    lags. It is disabled by default pending ablation.
    """

    def __init__(self, lags: Sequence[int] = (1, 2, 4, 8, 16, 32), eps: float = 1e-8):
        super().__init__()
        self.lags = tuple(int(lag) for lag in lags)
        self.eps = eps
        if any(lag <= 0 for lag in self.lags):
            raise ValueError("Cyclic-statistics lags must be positive")

    def _statistics(self, cycles: torch.Tensor) -> list[torch.Tensor]:
        values = normalized_cycles(cycles, self.eps)
        length = values.shape[-1]
        return [
            (values[..., :-lag] * values[..., lag:]).mean(dim=1)
            for lag in self.lags
            if lag < length
        ]

    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        estimate_statistics = self._statistics(estimate)
        target_statistics = self._statistics(target)
        if not estimate_statistics:
            return estimate.sum() * 0.0
        return torch.stack(
            [F.smooth_l1_loss(left, right) for left, right in zip(estimate_statistics, target_statistics)]
        ).mean()

