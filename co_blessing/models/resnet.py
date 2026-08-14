from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


NODE_CHANNELS = {"A": 64, "B": 64, "C": 128, "D": 256, "E": 512}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = F.relu(self.bn1(self.conv1(inputs)), inplace=False)
        output = self.bn2(self.conv2(output))
        output = F.relu(output + self.shortcut(inputs), inplace=False)
        return output


MaskMap = Mapping[str, Sequence[int] | torch.Tensor]


class CifarResNet(nn.Module):
    """CIFAR ResNet with the paper's five observable activation nodes."""

    def __init__(self, block: type[BasicBlock], blocks: Sequence[int], num_classes: int = 10) -> None:
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self, block: type[BasicBlock], planes: int, count: int, stride: int
    ) -> nn.Sequential:
        strides = [stride] + [1] * (count - 1)
        layers: list[nn.Module] = []
        for current_stride in strides:
            layers.append(block(self.in_planes, planes, current_stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    @staticmethod
    def _mask_channels(
        activation: torch.Tensor,
        channels: Sequence[int] | torch.Tensor | None,
    ) -> torch.Tensor:
        if channels is None:
            return activation
        index = torch.as_tensor(channels, dtype=torch.long, device=activation.device)
        if index.numel() == 0:
            return activation
        if index.min().item() < 0 or index.max().item() >= activation.shape[1]:
            raise IndexError(
                f"Mask contains a channel outside [0, {activation.shape[1] - 1}]"
            )
        # Avoid in-place writes: the unmasked activation may also participate in
        # the feature-difference loss in the same graph.
        keep = torch.ones(activation.shape[1], device=activation.device, dtype=activation.dtype)
        keep[index] = 0
        return activation * keep.view(1, -1, 1, 1)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        return_features: bool = False,
        masks: MaskMap | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        masks = masks or {}
        features: dict[str, torch.Tensor] = {}

        output = F.relu(self.bn1(self.conv1(inputs)), inplace=False)
        output = self._mask_channels(output, masks.get("A"))
        features["A"] = output

        output = self.layer1(output)
        output = self._mask_channels(output, masks.get("B"))
        features["B"] = output

        output = self.layer2(output)
        output = self._mask_channels(output, masks.get("C"))
        features["C"] = output

        output = self.layer3(output)
        output = self._mask_channels(output, masks.get("D"))
        features["D"] = output

        output = self.layer4(output)
        output = self._mask_channels(output, masks.get("E"))
        features["E"] = output

        output = F.avg_pool2d(output, 4)
        logits = self.linear(output.view(output.size(0), -1))
        if return_features:
            return logits, features
        return logits


def ResNet18(num_classes: int = 10) -> CifarResNet:
    return CifarResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
