from __future__ import annotations

import torch
from torch import nn


class CardiacConsensusModule(nn.Module):
    """Learn beat reliability and fuse a non-uniform cardiac consensus.

    Reliability is estimated from each beat's pooled feature, the global
    context, and their absolute discrepancy. Softmax weights sum to one across
    ``K`` and remain available for scientific inspection.
    """

    def __init__(self, channels: int, hidden_channels: int | None = None, dropout: float = 0.1):
        super().__init__()
        hidden = hidden_channels or max(channels // 2, 8)
        self.reliability = nn.Sequential(
            nn.Linear(channels * 3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.proposal = nn.Conv1d(channels * 2, channels, kernel_size=1)
        self.gate = nn.Conv1d(channels * 2, channels, kernel_size=1)
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 4:
            raise ValueError("Expected features with shape [B, K, C, P]")
        batch, cycles, channels, phases = features.shape
        pooled = features.mean(dim=-1)  # [B, K, C]
        context = pooled.mean(dim=1, keepdim=True).expand(-1, cycles, -1)
        scores = self.reliability(torch.cat((pooled, context, (pooled - context).abs()), dim=-1))
        weights = torch.softmax(scores.squeeze(-1), dim=1)  # [B, K], sum_K = 1
        consensus = torch.sum(features * weights[:, :, None, None], dim=1)  # [B, C, P]
        repeated = consensus[:, None].expand(-1, cycles, -1, -1)
        fusion_input = torch.cat((features, repeated), dim=2).reshape(
            batch * cycles, channels * 2, phases
        )
        proposal = torch.tanh(self.proposal(fusion_input))
        gate = torch.sigmoid(self.gate(fusion_input))
        original = features.reshape(batch * cycles, channels, phases)
        fused = self.norm(original + gate * (proposal - original))
        return fused.reshape(batch, cycles, channels, phases), weights

