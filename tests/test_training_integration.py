from __future__ import annotations

import copy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
from torch.utils.data import DataLoader, TensorDataset

import co_blessing.training as training
from co_blessing.config import DEFAULT_CONFIG
from co_blessing.data import LoaderBundle


class TinyFeatureModel(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.linear = nn.Linear(4, num_classes)

    def forward(self, inputs, *, return_features=False, masks=None):
        feature = torch.relu(self.conv(inputs))
        masks = masks or {}
        if "B" in masks and len(masks["B"]):
            keep = torch.ones(feature.shape[1], device=feature.device)
            keep[torch.as_tensor(masks["B"], dtype=torch.long)] = 0
            feature = feature * keep.view(1, -1, 1, 1)
        logits = self.linear(feature.mean(dim=(2, 3)))
        features = {node: feature for node in "ABCDE"}
        return (logits, features) if return_features else logits


def _loaders(seed: int) -> LoaderBundle:
    images = torch.rand(4, 3, 32, 32)
    targets = torch.tensor([0, 1, 2, 3])
    indexes = torch.arange(4)
    dataset = TensorDataset(images, targets, indexes)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True, generator=generator)
    test_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    return LoaderBundle(train_loader, test_loader, 4, 4, generator)


def test_two_epoch_training_and_resume(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(training, "ResNet18", TinyFeatureModel)
    monkeypatch.setattr(training, "build_cifar10_loaders", lambda _config, seed: _loaders(seed))
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["name"] = "smoke"
    config["device"] = "cpu"
    config["output"]["root"] = str(tmp_path)
    config["train"].update(
        {
            "objective": "ours_fd",
            "backend": "rs",
            "epochs": 1,
            "epsilon": 8,
            "monitor_pgd_steps": 1,
            "monitor_subset": 2,
            "track_features": True,
        }
    )
    run = training.train(config)
    assert (run / "best.pt").exists()
    assert (run / "final.pt").exists()
    assert (run / "resume.pt").exists()

    resumed = copy.deepcopy(config)
    resumed["train"]["epochs"] = 2
    training.train(resumed, resume=str(run / "resume.pt"))
    checkpoint = torch.load(run / "final.pt", map_location="cpu")
    assert checkpoint["epoch"] == 1
