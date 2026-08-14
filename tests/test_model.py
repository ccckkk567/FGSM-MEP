from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from co_blessing.models import ResNet18


def test_resnet_exposes_paper_nodes_and_shapes() -> None:
    model = ResNet18()
    inputs = torch.randn(2, 3, 32, 32, requires_grad=True)
    logits, features = model(inputs, return_features=True)
    assert logits.shape == (2, 10)
    assert {key: value.shape for key, value in features.items()} == {
        "A": (2, 64, 32, 32),
        "B": (2, 64, 32, 32),
        "C": (2, 128, 16, 16),
        "D": (2, 256, 8, 8),
        "E": (2, 512, 4, 4),
    }
    logits.sum().backward()
    assert inputs.grad is not None


def test_channel_mask_is_zero_and_affects_downstream() -> None:
    model = ResNet18().eval()
    inputs = torch.randn(1, 3, 32, 32)
    plain, _ = model(inputs, return_features=True)
    masked, features = model(inputs, return_features=True, masks={"A": [0, 3]})
    assert torch.count_nonzero(features["A"][:, [0, 3]]) == 0
    assert not torch.allclose(plain, masked)
