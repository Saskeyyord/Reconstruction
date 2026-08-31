from __future__ import annotations

import torch
from torch import nn


class CrossCycleAttention(nn.Module):
    """Axial self-attention across cardiac cycles at each latent phase.

    Input/output shape is ``[B, K, C, P]``. Motion may be periodic in absolute
    time, but components that are not consistently phase-locked to the cardiac
    cycle tend to be less coherent across the ``K`` tokens at fixed phase ``P``.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        feedforward_multiplier: int = 2,
    ):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(channels)
        hidden = channels * feedforward_multiplier
        self.feedforward = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError("Expected features with shape [B, K, C, P]")
        batch, cycles, channels, phases = features.shape
        # [B, K, C, P] -> [B*P, K, C]: attention only along cycle axis K.
        tokens = features.permute(0, 3, 1, 2).reshape(batch * phases, cycles, channels)
        normalized = self.norm1(tokens)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + self.dropout1(attended)
        tokens = tokens + self.dropout2(self.feedforward(self.norm2(tokens)))
        return tokens.reshape(batch, phases, cycles, channels).permute(0, 2, 3, 1)

