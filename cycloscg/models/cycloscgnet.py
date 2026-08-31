from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .baseline_unet import ResidualConvBlock1D
from .consensus import CardiacConsensusModule
from .cross_cycle_attention import CrossCycleAttention


@dataclass(frozen=True)
class CycloSCGOutput:
    reconstruction: torch.Tensor
    reliability_weights: torch.Tensor


class SharedBeatEncoder(nn.Module):
    def __init__(self, base_channels: int, dropout: float):
        super().__init__()
        self.stem = ResidualConvBlock1D(1, base_channels, dropout)
        self.down1 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            ResidualConvBlock1D(base_channels * 2, base_channels * 2, dropout),
        )
        self.down2 = nn.Sequential(
            nn.Conv1d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1),
            ResidualConvBlock1D(base_channels * 4, base_channels * 4, dropout),
        )

    def forward(self, beats: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # beats: [B*K, 1, L]
        level0 = self.stem(beats)
        level1 = self.down1(level0)
        latent = self.down2(level1)
        return latent, [level0, level1]


class SharedBeatDecoder(nn.Module):
    def __init__(self, base_channels: int, dropout: float):
        super().__init__()
        self.up1 = nn.ConvTranspose1d(base_channels * 4, base_channels * 2, 4, 2, 1)
        self.fuse1 = ResidualConvBlock1D(base_channels * 4, base_channels * 2, dropout)
        self.up2 = nn.ConvTranspose1d(base_channels * 2, base_channels, 4, 2, 1)
        self.fuse2 = ResidualConvBlock1D(base_channels * 2, base_channels, dropout)
        self.output = nn.Conv1d(base_channels, 1, kernel_size=1)

    def forward(self, latent: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        x = self.up1(latent)
        if x.shape[-1] != skips[1].shape[-1]:
            x = F.interpolate(x, size=skips[1].shape[-1], mode="linear", align_corners=False)
        x = self.fuse1(torch.cat((x, skips[1]), dim=1))
        x = self.up2(x)
        if x.shape[-1] != skips[0].shape[-1]:
            x = F.interpolate(x, size=skips[0].shape[-1], mode="linear", align_corners=False)
        return self.output(self.fuse2(torch.cat((x, skips[0]), dim=1)))


class CycloSCGNet(nn.Module):
    """Cyclostationarity-guided SCG reconstruction network.

    The network consumes cardiac-phase matrices ``[B, K, L]``. A shared beat
    encoder avoids beat-specific parameters; axial attention and learnable
    consensus exchange information across cycles before shared decoding.
    """

    def __init__(
        self,
        base_channels: int = 16,
        attention_heads: int = 4,
        dropout: float = 0.1,
        use_cross_cycle_attention: bool = True,
        use_consensus: bool = True,
        residual_reconstruction: bool = True,
    ):
        super().__init__()
        self.use_cross_cycle_attention = bool(use_cross_cycle_attention)
        self.use_consensus = bool(use_consensus)
        self.residual_reconstruction = bool(residual_reconstruction)
        self.encoder = SharedBeatEncoder(base_channels, dropout)
        latent_channels = base_channels * 4
        self.cross_cycle = CrossCycleAttention(
            latent_channels, num_heads=attention_heads, dropout=dropout
        )
        self.consensus = CardiacConsensusModule(latent_channels, dropout=dropout)
        self.decoder = SharedBeatDecoder(base_channels, dropout)
        if self.residual_reconstruction:
            # Start from identity reconstruction. This protects clean morphology
            # before the residual artifact-correction branch has learned.
            nn.init.zeros_(self.decoder.output.weight)
            nn.init.zeros_(self.decoder.output.bias)

    def forward(self, cycle_matrix: torch.Tensor, return_aux: bool = False):
        if cycle_matrix.ndim != 3:
            raise ValueError("Expected cycle matrix with shape [B, K, L]")
        batch, cycles, phase_length = cycle_matrix.shape
        pad_right = (-phase_length) % 4
        padded = F.pad(cycle_matrix, (0, pad_right), mode="replicate") if pad_right else cycle_matrix
        flattened = padded.reshape(batch * cycles, 1, padded.shape[-1])
        latent_flat, skips = self.encoder(flattened)
        channels, latent_phase = latent_flat.shape[1:]
        latent = latent_flat.reshape(batch, cycles, channels, latent_phase)
        if self.use_cross_cycle_attention:
            latent = self.cross_cycle(latent)
        if self.use_consensus:
            latent, reliability = self.consensus(latent)
        else:
            reliability = torch.full(
                (batch, cycles),
                1.0 / cycles,
                dtype=latent.dtype,
                device=latent.device,
            )
        decoded = self.decoder(latent.reshape(batch * cycles, channels, latent_phase), skips)
        decoded = decoded.reshape(batch, cycles, padded.shape[-1])[..., :phase_length]
        reconstruction = cycle_matrix + decoded if self.residual_reconstruction else decoded
        if return_aux:
            return CycloSCGOutput(reconstruction, reliability)
        return reconstruction
