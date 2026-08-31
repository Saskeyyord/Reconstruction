import torch

from cycloscg.losses import (
    CrossCycleCoherenceLoss,
    CyclicStatisticsLoss,
    MultiResolutionSTFTLoss,
    PhaseCovarianceLoss,
    SingularSpectrumLoss,
    WaveformLoss,
)


def test_all_losses_are_finite_and_differentiable() -> None:
    estimate = torch.randn(2, 6, 256, requires_grad=True)
    target = torch.randn_like(estimate)
    losses = [
        WaveformLoss()(estimate, target),
        CrossCycleCoherenceLoss()(estimate, target),
        PhaseCovarianceLoss()(estimate, target),
        SingularSpectrumLoss()(estimate, target),
        MultiResolutionSTFTLoss()(estimate, target),
        CyclicStatisticsLoss()(estimate, target),
    ]
    assert all(torch.isfinite(value) for value in losses)
    total = torch.stack(losses).sum()
    total.backward()
    assert estimate.grad is not None
    assert torch.isfinite(estimate.grad).all()

