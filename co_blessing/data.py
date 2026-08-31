from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


class ReferenceCIFARAugment:
    """Pad/crop/flip in the same order and RNG family as FGSM-PGI."""

    def __init__(self) -> None:
        self.pad_and_tensor = transforms.Compose([transforms.Pad(4), transforms.ToTensor()])

    def __call__(self, image: Any) -> torch.Tensor:
        padded = self.pad_and_tensor(image)
        flipped = bool(random.getrandbits(1))
        crop_x = random.randint(0, 8)
        crop_y = random.randint(0, 8)
        output = padded[:, crop_x : crop_x + 32, crop_y : crop_y + 32]
        if flipped:
            output = torch.flip(output, dims=(2,))
        return output


def _train_transform(config: dict[str, Any]) -> transforms.Compose | ReferenceCIFARAugment:
    augmentation = str(config.get("augmentation", "fgsm_mep")).lower()
    if augmentation == "fgsm_mep":
        return ReferenceCIFARAugment()
    if augmentation == "aaer":
        # Keep normalization out of the dataset: InputNormalizedModel applies
        # the official AAER constants inside the model, so MEP's stored delta
        # and every attack remain expressed in raw pixel coordinates.
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    raise ValueError(f"Unsupported augmentation: {augmentation}")


class IndexedDataset(Dataset[tuple[torch.Tensor, int, int]]):
    """Attach stable sample IDs required by FGSM-MEP state buffers."""

    def __init__(self, dataset: Dataset[tuple[torch.Tensor, int]]) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        image, target = self.dataset[index]
        return image, target, index


@dataclass(frozen=True)
class LoaderBundle:
    train: DataLoader[Any]
    test: DataLoader[Any]
    train_size: int
    test_size: int
    train_generator: torch.Generator


def _subset(dataset: Dataset[Any], count: int | None) -> Dataset[Any]:
    if count is None:
        return dataset
    if count <= 0:
        raise ValueError("Subset size must be positive")
    return Subset(dataset, range(min(count, len(dataset))))


def _build_cifar_loaders(
    dataset_class: type[datasets.CIFAR10] | type[datasets.CIFAR100],
    config: dict[str, Any],
    seed: int,
) -> LoaderBundle:
    root = Path(config["root"]).expanduser()
    train_transform = _train_transform(config)
    test_transform = transforms.ToTensor()
    train_base = dataset_class(
        str(root), train=True, transform=train_transform, download=bool(config["download"])
    )
    test_base = dataset_class(
        str(root), train=False, transform=test_transform, download=bool(config["download"])
    )
    train_dataset = _subset(IndexedDataset(train_base), config.get("train_subset"))
    test_dataset = _subset(IndexedDataset(test_base), config.get("test_subset"))

    generator = torch.Generator()
    generator.manual_seed(seed)
    common = {
        "batch_size": int(config["batch_size"]),
        "num_workers": int(config["num_workers"]),
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": int(config["num_workers"]) > 0,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, drop_last=False, **common
    )
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **common)
    return LoaderBundle(
        train=train_loader,
        test=test_loader,
        train_size=len(train_dataset),
        test_size=len(test_dataset),
        train_generator=generator,
    )


def build_cifar10_loaders(config: dict[str, Any], seed: int) -> LoaderBundle:
    return _build_cifar_loaders(datasets.CIFAR10, config, seed)


def build_cifar100_loaders(config: dict[str, Any], seed: int) -> LoaderBundle:
    return _build_cifar_loaders(datasets.CIFAR100, config, seed)


def build_cifar_loaders(config: dict[str, Any], seed: int) -> LoaderBundle:
    dataset = str(config.get("dataset", "")).lower()
    if dataset == "cifar10":
        return build_cifar10_loaders(config, seed)
    if dataset == "cifar100":
        return build_cifar100_loaders(config, seed)
    raise ValueError(f"Unsupported dataset: {dataset}")
