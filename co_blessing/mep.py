from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .attacks import project_linf


@dataclass
class TrainingPerturbation:
    initial: torch.Tensor
    adversarial: torch.Tensor
    next_prior: torch.Tensor | None
    next_momentum: torch.Tensor | None


class MEPState:
    """Per-example perturbation and momentum buffers from FGSM-PGI/FGSM-MEP."""

    def __init__(
        self,
        sample_count: int,
        image_shape: tuple[int, int, int],
        epsilon: float,
        alpha: float,
        device: torch.device,
    ) -> None:
        self.sample_count = sample_count
        self.image_shape = image_shape
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.device = device
        shape = (sample_count, *image_shape)
        self.delta = torch.empty(shape, dtype=torch.float32, device=device)
        self.momentum = torch.zeros(shape, dtype=torch.float32, device=device)
        self.last_reset_epoch = -1

    def reset(self, epoch: int) -> None:
        self.delta.uniform_(-self.epsilon, self.epsilon)
        # The reference implementation maps the uniform draw to random corners
        # with alpha * sign and then clips it to epsilon.
        self.delta.copy_(torch.clamp(self.alpha * self.delta.sign(), -self.epsilon, self.epsilon))
        self.momentum.zero_()
        self.last_reset_epoch = int(epoch)

    def get(self, sample_ids: torch.Tensor, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        index = sample_ids.to(self.device, non_blocking=True, dtype=torch.long)
        delta = self.delta.index_select(0, index).detach().clone()
        momentum = self.momentum.index_select(0, index).detach().clone()
        # FGSM-PGI feeds the stored prior directly into the initial forward.
        # It is projected against the current augmented image only when the
        # current and next adversarial deltas are updated.
        return delta, momentum

    @torch.no_grad()
    def update(
        self,
        sample_ids: torch.Tensor,
        next_delta: torch.Tensor,
        next_momentum: torch.Tensor,
    ) -> None:
        index = sample_ids.to(self.device, non_blocking=True, dtype=torch.long)
        self.delta.index_copy_(0, index, next_delta.detach())
        self.momentum.index_copy_(0, index, next_momentum.detach())

    def state_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "image_shape": self.image_shape,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "last_reset_epoch": self.last_reset_epoch,
            "delta": self.delta.detach().cpu(),
            "momentum": self.momentum.detach().cpu(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = (self.sample_count, *self.image_shape)
        if tuple(state["delta"].shape) != expected or tuple(state["momentum"].shape) != expected:
            raise ValueError(f"MEP checkpoint shape does not match {expected}")
        if abs(float(state["epsilon"]) - self.epsilon) > 1e-12:
            raise ValueError("MEP checkpoint epsilon differs from the current config")
        self.delta.copy_(state["delta"].to(self.device))
        self.momentum.copy_(state["momentum"].to(self.device))
        self.last_reset_epoch = int(state["last_reset_epoch"])


def build_training_perturbation(
    *,
    initial: torch.Tensor,
    inputs: torch.Tensor,
    input_gradient: torch.Tensor,
    epsilon: float,
    alpha: float,
    previous_momentum: torch.Tensor | None,
    momentum_decay: float,
) -> TrainingPerturbation:
    global_l1 = input_gradient.abs().sum().clamp_min(1e-12)
    if previous_momentum is None:
        momentum = input_gradient / global_l1
        next_prior = None
    else:
        momentum = input_gradient / global_l1 + momentum_decay * previous_momentum
        next_prior = project_linf(initial + alpha * momentum.sign(), inputs, epsilon).detach()
    adversarial = project_linf(initial + alpha * input_gradient.sign(), inputs, epsilon).detach()
    return TrainingPerturbation(
        initial=initial,
        adversarial=adversarial,
        next_prior=next_prior,
        next_momentum=momentum.detach() if previous_momentum is not None else None,
    )


def random_start(inputs: torch.Tensor, epsilon: float) -> torch.Tensor:
    delta = torch.empty_like(inputs).uniform_(-epsilon, epsilon)
    return project_linf(delta, inputs, epsilon).detach()
