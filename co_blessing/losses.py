from __future__ import annotations

import torch
import torch.nn.functional as F


def smooth_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    true_probability: float = 0.6,
) -> torch.Tensor:
    """FGSM-MEP's exact target distribution, not PyTorch's smoothing convention."""

    classes = logits.shape[1]
    if classes <= 1:
        raise ValueError("At least two classes are required")
    if not 0 < true_probability <= 1:
        raise ValueError("true_probability must be in (0, 1]")
    off_probability = (1.0 - true_probability) / (classes - 1)
    distribution = torch.full_like(logits, off_probability)
    distribution.scatter_(1, targets[:, None], true_probability)
    return -(distribution * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def feature_difference(
    adversarial: torch.Tensor,
    initial: torch.Tensor,
    channels: torch.Tensor | list[int] | None = None,
) -> torch.Tensor:
    if channels is not None:
        index = torch.as_tensor(channels, dtype=torch.long, device=adversarial.device)
        if index.numel() == 0:
            return adversarial.new_zeros(())
        adversarial = adversarial.index_select(1, index)
        initial = initial.index_select(1, index)
    return (adversarial - initial).square().mean()


def channel_difference_scores(
    adversarial: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Mean squared activation difference per channel."""

    return (adversarial - reference).square().mean(dim=(0, 2, 3))
