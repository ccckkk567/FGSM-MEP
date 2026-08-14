from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AttackTrace:
    delta: torch.Tensor
    steps_run: int


def project_linf(delta: torch.Tensor, inputs: torch.Tensor, epsilon: float) -> torch.Tensor:
    delta = delta.clamp(-epsilon, epsilon)
    return (inputs + delta).clamp(0.0, 1.0) - inputs


def fgsm(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    delta = torch.zeros_like(inputs, requires_grad=True)
    logits = model(inputs + delta)
    gradient = torch.autograd.grad(F.cross_entropy(logits, targets), delta)[0]
    return project_linf(epsilon * gradient.sign(), inputs, epsilon).detach()


def pgd(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float,
    step_size: float,
    steps: int,
    restarts: int = 1,
    *,
    random_start: bool = True,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    return_trace: bool = False,
) -> torch.Tensor | AttackTrace:
    """Full-iteration L-inf PGD without correctness-based early stopping."""

    if steps <= 0 or restarts <= 0:
        raise ValueError("PGD steps and restarts must be positive")
    loss_fn = loss_fn or F.cross_entropy
    max_loss = torch.full((inputs.shape[0],), -torch.inf, device=inputs.device)
    max_delta = torch.zeros_like(inputs)
    steps_run = 0
    for _ in range(restarts):
        if random_start:
            delta = torch.empty_like(inputs).uniform_(-epsilon, epsilon)
            delta = project_linf(delta, inputs, epsilon)
        else:
            delta = torch.zeros_like(inputs)
        for _ in range(steps):
            delta.requires_grad_(True)
            logits = model(inputs + delta)
            loss = loss_fn(logits, targets)
            gradient = torch.autograd.grad(loss, delta)[0]
            delta = project_linf(delta.detach() + step_size * gradient.sign(), inputs, epsilon)
            steps_run += 1
        with torch.no_grad():
            logits = model(inputs + delta)
            losses = F.cross_entropy(logits, targets, reduction="none")
            replace = losses >= max_loss
            max_loss = torch.where(replace, losses, max_loss)
            max_delta[replace] = delta.detach()[replace]
    if return_trace:
        return AttackTrace(max_delta, steps_run)
    return max_delta


def cw_margin(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    true_logits = logits.gather(1, targets[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, targets[:, None], -torch.inf)
    other_logits = masked.max(dim=1).values
    return (other_logits - true_logits).mean()


def cw_linf(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float,
    step_size: float,
    steps: int = 20,
    restarts: int = 1,
    *,
    return_trace: bool = False,
) -> torch.Tensor | AttackTrace:
    return pgd(
        model,
        inputs,
        targets,
        epsilon,
        step_size,
        steps,
        restarts,
        random_start=True,
        loss_fn=cw_margin,
        return_trace=return_trace,
    )


def parse_attack(name: str) -> tuple[str, int | None]:
    normalized = name.lower()
    if normalized in {"clean", "fgsm", "apgd-t", "aa"}:
        return normalized, None
    for prefix in ("pgd", "cw"):
        if normalized.startswith(prefix):
            suffix = normalized[len(prefix) :]
            if not suffix.isdigit():
                raise ValueError(f"Attack requires an integer step suffix: {name}")
            return prefix, int(suffix)
    raise ValueError(f"Unknown attack: {name}")
