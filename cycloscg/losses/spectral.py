from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class MultiResolutionSTFTLoss(nn.Module):
    """Auxiliary multi-resolution STFT magnitude fidelity loss (not ConceFT)."""

    def __init__(
        self,
        resolutions: Sequence[Sequence[int]] = ((64, 16, 64), (128, 32, 128), (256, 64, 256)),
        eps: float = 1e-7,
    ):
        super().__init__()
        self.resolutions = tuple(tuple(map(int, values)) for values in resolutions)
        if any(len(values) != 3 for values in self.resolutions):
            raise ValueError("Each STFT resolution must be [n_fft, hop_length, win_length]")
        self.eps = eps

    def _magnitude(self, waveforms: torch.Tensor, resolution: tuple[int, int, int]) -> torch.Tensor:
        n_fft, hop_length, win_length = resolution
        if waveforms.shape[-1] < n_fft:
            waveforms = F.pad(waveforms, (0, n_fft - waveforms.shape[-1]))
        window = torch.hann_window(win_length, device=waveforms.device, dtype=waveforms.dtype)
        transform = torch.stft(
            waveforms,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        return transform.abs().clamp_min(self.eps)

    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        estimate_flat = estimate.reshape(-1, estimate.shape[-1])
        target_flat = target.reshape(-1, target.shape[-1])
        losses: list[torch.Tensor] = []
        for resolution in self.resolutions:
            estimate_magnitude = self._magnitude(estimate_flat, resolution)
            target_magnitude = self._magnitude(target_flat, resolution)
            spectral_convergence = torch.linalg.vector_norm(estimate_magnitude - target_magnitude) / (
                torch.linalg.vector_norm(target_magnitude).clamp_min(self.eps)
            )
            log_magnitude = torch.mean(
                torch.abs(torch.log(estimate_magnitude) - torch.log(target_magnitude))
            )
            losses.append(spectral_convergence + log_magnitude)
        return torch.stack(losses).mean()
