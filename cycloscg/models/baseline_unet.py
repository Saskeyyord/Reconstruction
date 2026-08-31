from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualConvBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)


class ResidualUNet1D(nn.Module):
    """Conventional direct-waveform U-Net baseline.

    This model intentionally operates on ``[B, T]`` waveforms and contains no
    cardiac-phase or cross-cycle modeling. It is a capacity-controlled baseline
    for testing whether explicit cyclostationary structure adds value.
    """

    def __init__(
        self,
        base_channels: int = 16,
        depth: int = 3,
        dropout: float = 0.05,
        residual_reconstruction: bool = True,
    ):
        super().__init__()
        if base_channels <= 0 or depth < 1:
            raise ValueError("base_channels must be positive and depth at least one")
        self.depth = int(depth)
        self.residual_reconstruction = bool(residual_reconstruction)
        channels = [base_channels * (2**level) for level in range(depth + 1)]
        self.stem = ResidualConvBlock1D(1, channels[0], dropout)
        self.downsamples = nn.ModuleList()
        for level in range(depth):
            self.downsamples.append(
                nn.Sequential(
                    nn.Conv1d(
                        channels[level], channels[level + 1], kernel_size=4, stride=2, padding=1
                    ),
                    ResidualConvBlock1D(channels[level + 1], channels[level + 1], dropout),
                )
            )
        self.bottleneck = ResidualConvBlock1D(channels[-1], channels[-1], dropout)
        self.upsamples = nn.ModuleList()
        for level in reversed(range(depth)):
            self.upsamples.append(
                nn.ModuleDict(
                    {
                        "up": nn.ConvTranspose1d(
                            channels[level + 1],
                            channels[level],
                            kernel_size=4,
                            stride=2,
                            padding=1,
                        ),
                        "fuse": ResidualConvBlock1D(
                            channels[level] * 2, channels[level], dropout
                        ),
                    }
                )
            )
        self.output = nn.Conv1d(channels[0], 1, kernel_size=1)
        if self.residual_reconstruction:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        squeeze_channel = waveform.ndim == 2
        if squeeze_channel:
            waveform = waveform.unsqueeze(1)
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError("Expected waveform with shape [B, T] or [B, 1, T]")

        original = waveform
        divisor = 2**self.depth
        pad_right = (-waveform.shape[-1]) % divisor
        if pad_right:
            waveform = F.pad(waveform, (0, pad_right), mode="replicate")

        skips: list[torch.Tensor] = []
        x = self.stem(waveform)
        skips.append(x)
        for downsample in self.downsamples:
            x = downsample(x)
            skips.append(x)
        x = self.bottleneck(x)

        for decoder_index, modules in enumerate(self.upsamples):
            x = modules["up"](x)
            skip = skips[-decoder_index - 2]
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
            x = modules["fuse"](torch.cat((x, skip), dim=1))
        delta_or_output = self.output(x)
        if self.residual_reconstruction:
            padded_original = waveform
            reconstructed = padded_original + delta_or_output
        else:
            reconstructed = delta_or_output
        reconstructed = reconstructed[..., : original.shape[-1]]
        return reconstructed.squeeze(1) if squeeze_channel else reconstructed
