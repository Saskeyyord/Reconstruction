import torch

from cycloscg.models import CardiacConsensusModule, CrossCycleAttention, CycloSCGNet


def test_cross_cycle_attention_preserves_shape() -> None:
    module = CrossCycleAttention(channels=16, num_heads=4, dropout=0.0)
    features = torch.randn(2, 12, 16, 32)
    output = module(features)
    assert output.shape == features.shape
    assert torch.isfinite(output).all()


def test_consensus_weights_sum_to_one() -> None:
    module = CardiacConsensusModule(channels=16, dropout=0.0)
    features = torch.randn(3, 12, 16, 32)
    fused, weights = module(features)
    assert fused.shape == features.shape
    assert weights.shape == (3, 12)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(3), atol=1e-6, rtol=1e-6)


def test_cycloscgnet_minibatch_forward_backward() -> None:
    model = CycloSCGNet(base_channels=4, attention_heads=4, dropout=0.0)
    noisy = torch.randn(2, 6, 128)
    target = torch.randn_like(noisy)
    output = model(noisy, return_aux=True)
    assert output.reconstruction.shape == noisy.shape
    assert output.reliability_weights.shape == (2, 6)
    loss = torch.nn.functional.smooth_l1_loss(output.reconstruction, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())

