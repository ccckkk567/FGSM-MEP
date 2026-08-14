from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from co_blessing.mep import MEPState, build_training_perturbation


def test_mep_state_is_indexed_reset_and_restorable() -> None:
    epsilon = 8 / 255
    state = MEPState(4, (3, 4, 4), epsilon, epsilon, torch.device("cpu"))
    state.reset(0)
    assert torch.allclose(state.delta.abs(), torch.full_like(state.delta, epsilon))
    original = state.delta.clone()
    ids = torch.tensor([1, 3])
    replacement = torch.zeros(2, 3, 4, 4)
    momentum = torch.ones_like(replacement)
    state.update(ids, replacement, momentum)
    assert torch.equal(state.delta[0], original[0])
    assert torch.equal(state.delta[1], replacement[0])

    restored = MEPState(4, (3, 4, 4), epsilon, epsilon, torch.device("cpu"))
    restored.load_state_dict(state.state_dict())
    assert torch.equal(restored.delta, state.delta)
    assert torch.equal(restored.momentum, state.momentum)


def test_mep_update_uses_global_l1_momentum_and_bounds() -> None:
    inputs = torch.full((2, 3, 4, 4), 0.5)
    initial = torch.zeros_like(inputs)
    gradient = torch.ones_like(inputs)
    previous = torch.zeros_like(inputs)
    epsilon = 8 / 255
    result = build_training_perturbation(
        initial=initial,
        inputs=inputs,
        input_gradient=gradient,
        epsilon=epsilon,
        alpha=epsilon,
        previous_momentum=previous,
        momentum_decay=0.3,
    )
    expected_momentum = torch.full_like(gradient, 1 / gradient.numel())
    assert torch.allclose(result.next_momentum, expected_momentum)
    assert result.adversarial.abs().max() <= epsilon + 1e-7
    assert result.next_prior.abs().max() <= epsilon + 1e-7
