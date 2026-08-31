import torch

from cycloscg.models.baseline_unet import ResidualUNet1D


def test_baseline_forward_backward_preserves_shape() -> None:
    model = ResidualUNet1D(base_channels=4, depth=2, dropout=0.0)
    noisy = torch.randn(2, 513, requires_grad=True)
    clean = torch.randn_like(noisy)
    reconstructed = model(noisy)
    assert reconstructed.shape == noisy.shape
    loss = torch.nn.functional.smooth_l1_loss(reconstructed, clean)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

