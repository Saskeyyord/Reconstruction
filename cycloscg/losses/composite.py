from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .cycle_coherence import CrossCycleCoherenceLoss
from .cyclostationary import CyclicStatisticsLoss, PhaseCovarianceLoss
from .singular_spectrum import SingularSpectrumLoss
from .spectral import MultiResolutionSTFTLoss
from .waveform import WaveformLoss


class CycloSCGLoss(nn.Module):
    def __init__(
        self,
        weights: Mapping[str, float] | None = None,
        stft_resolutions: Sequence[Sequence[int]] = ((64, 16, 64), (128, 32, 128), (256, 64, 256)),
        cyclic_lags: Sequence[int] = (1, 2, 4, 8, 16, 32),
        use_cyclic_loss: bool = False,
    ):
        super().__init__()
        defaults = {
            "wave": 1.0,
            "cycle": 0.2,
            "cov": 0.1,
            "svd": 0.1,
            "spec": 0.05,
            "cyclic": 0.0,
            "identity": 0.2,
        }
        self.weights = {**defaults, **dict(weights or {})}
        self.use_cyclic_loss = bool(use_cyclic_loss)
        self.waveform = WaveformLoss("smooth_l1")
        self.cycle = CrossCycleCoherenceLoss()
        self.covariance = PhaseCovarianceLoss()
        self.singular = SingularSpectrumLoss()
        self.spectral = MultiResolutionSTFTLoss(stft_resolutions)
        self.cyclic = CyclicStatisticsLoss(cyclic_lags)

    def forward(
        self,
        estimate: torch.Tensor,
        target: torch.Tensor,
        identity_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        components = {
            "wave": self.waveform(estimate, target),
            "cycle": self.cycle(estimate, target),
            "cov": self.covariance(estimate, target),
            "svd": self.singular(estimate, target),
            "spec": self.spectral(estimate, target),
        }
        if self.use_cyclic_loss:
            components["cyclic"] = self.cyclic(estimate, target)
        else:
            components["cyclic"] = estimate.sum() * 0.0
        if identity_mask is not None and torch.any(identity_mask):
            components["identity"] = F.l1_loss(estimate[identity_mask], target[identity_mask])
        else:
            components["identity"] = estimate.sum() * 0.0
        total = sum(self.weights[name] * value for name, value in components.items())
        return total, components

