from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from co_blessing.attacks import cw_linf, pgd
from co_blessing.autoattack_adapter import EXPECTED_SHA256, source_metadata
from co_blessing.evaluation import add_inference_noise
from co_blessing.losses import channel_difference_scores, feature_difference, smooth_cross_entropy


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3 * 4 * 4, 3)

    def forward(self, inputs):
        return self.linear(inputs.flatten(1))


def test_smooth_cross_entropy_matches_explicit_distribution() -> None:
    logits = torch.tensor([[2.0, 0.0, -1.0]])
    target = torch.tensor([0])
    expected_distribution = torch.tensor([[0.6, 0.2, 0.2]])
    expected = -(expected_distribution * logits.log_softmax(1)).sum()
    assert torch.allclose(smooth_cross_entropy(logits, target, 0.6), expected)


def test_feature_losses_and_channel_scores() -> None:
    reference = torch.zeros(2, 3, 2, 2)
    adversarial = reference.clone()
    adversarial[:, 1] = 2
    assert feature_difference(adversarial, reference).item() > 0
    assert feature_difference(adversarial, reference, [0]).item() == 0
    assert channel_difference_scores(adversarial, reference).argmax().item() == 1


def test_pgd_and_cw_are_bounded_and_run_all_steps() -> None:
    model = TinyClassifier()
    inputs = torch.rand(5, 3, 4, 4)
    targets = torch.randint(0, 3, (5,))
    epsilon = 8 / 255
    trace = pgd(
        model,
        inputs,
        targets,
        epsilon,
        2 / 255,
        steps=4,
        restarts=2,
        return_trace=True,
    )
    assert trace.steps_run == 8
    assert trace.delta.abs().max() <= epsilon + 1e-7
    assert (inputs + trace.delta).min() >= 0 and (inputs + trace.delta).max() <= 1
    cw_trace = cw_linf(
        model,
        inputs,
        targets,
        epsilon,
        2 / 255,
        steps=3,
        return_trace=True,
    )
    assert cw_trace.steps_run == 3
    assert cw_trace.delta.abs().max() <= epsilon + 1e-7


def test_inference_noise_is_seeded_and_applied_to_existing_examples() -> None:
    examples = torch.full((2, 3, 4, 4), 0.5)
    config = {"enabled": True, "kind": "uniform", "magnitude": 16, "seed": 0}
    generator1 = torch.Generator().manual_seed(7)
    generator2 = torch.Generator().manual_seed(7)
    first = add_inference_noise(examples, config, generator1)
    second = add_inference_noise(examples, config, generator2)
    assert torch.equal(first, second)
    assert not torch.equal(first, examples)
    assert first.min() >= 0 and first.max() <= 1


def test_reference_autoattack_snapshot_is_pinned() -> None:
    metadata = source_metadata()
    assert metadata["available"]
    assert metadata["matches_expected"]
    assert metadata["sha256"] == EXPECTED_SHA256
