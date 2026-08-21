from __future__ import annotations

import copy
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from .attacks import pgd
from .config import validate_config
from .data import build_cifar10_loaders
from .losses import channel_difference_scores, feature_difference, smooth_cross_entropy
from .mep import MEPState, build_training_perturbation, random_start
from .models import ResNet18
from .models.resnet import NODE_CHANNELS
from .utils import (
    append_csv,
    atomic_torch_save,
    environment_metadata,
    load_model_state,
    resolve_device,
    seed_everything,
    write_json,
)


TRAIN_FIELDS = [
    "epoch",
    "seconds",
    "lr",
    "train_loss",
    "train_accuracy",
    "monitor_clean_accuracy",
    "monitor_pgd10_accuracy",
    "vact_A",
    "vact_B",
    "vact_C",
    "vact_D",
    "vact_E",
    "selected_channels",
]


def _forward_features(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    masks: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    output = model(inputs, return_features=True, masks=masks)
    if not isinstance(output, tuple):
        raise TypeError("Model did not return feature maps")
    return output


@torch.no_grad()
def clean_accuracy(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        correct += logits.argmax(1).eq(targets).sum().item()
        total += targets.numel()
    return correct / max(total, 1)


def pgd_accuracy(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    epsilon: float,
    steps: int,
    subset: int | None = None,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, targets, _ in loader:
        if subset is not None and total >= subset:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if subset is not None and total + targets.numel() > subset:
            keep = subset - total
            images, targets = images[:keep], targets[:keep]
        delta = pgd(
            model,
            images,
            targets,
            epsilon=epsilon,
            step_size=2.0 / 255.0,
            steps=steps,
            restarts=1,
            random_start=True,
        )
        with torch.no_grad():
            logits = model(images + delta)
        correct += logits.argmax(1).eq(targets).sum().item()
        total += targets.numel()
    return correct / max(total, 1)


def pgd_monitor(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    epsilon: float,
    steps: int,
    subset: int | None,
    track_features: bool,
) -> tuple[float, dict[str, float] | None]:
    model.eval()
    correct = 0
    total = 0
    feature_sums = {node: 0.0 for node in NODE_CHANNELS}
    for images, targets, _ in loader:
        if subset is not None and total >= subset:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if subset is not None and total + targets.numel() > subset:
            keep = subset - total
            images, targets = images[:keep], targets[:keep]
        delta = pgd(
            model,
            images,
            targets,
            epsilon=epsilon,
            step_size=2.0 / 255.0,
            steps=steps,
            restarts=1,
            random_start=True,
        )
        with torch.no_grad():
            if track_features:
                _, clean_features = model(images, return_features=True)
                logits, adv_features = model(images + delta, return_features=True)
                for node in NODE_CHANNELS:
                    difference = adv_features[node] - clean_features[node]
                    feature_sums[node] += difference.flatten(1).norm(p=2, dim=1).sum().item()
            else:
                logits = model(images + delta)
        correct += logits.argmax(1).eq(targets).sum().item()
        total += targets.numel()
    vact = None
    if track_features:
        vact = {node: feature_sums[node] / max(total, 1) for node in NODE_CHANNELS}
    return correct / max(total, 1), vact


def _rng_state(loader_generator: torch.Generator) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "loader_generator": loader_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any], loader_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    loader_generator.set_state(state["loader_generator"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _light_checkpoint(
    model: torch.nn.Module,
    config: dict[str, Any],
    epoch: int,
    metrics: dict[str, float],
    selected_channels: torch.Tensor,
) -> dict[str, Any]:
    return {
        "model": copy.deepcopy(model.state_dict()),
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
        "epoch": epoch,
        "metrics": metrics,
        "selected_channels": selected_channels.detach().cpu(),
    }


def _resume_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    mep_state: MEPState | None,
    config: dict[str, Any],
    epoch: int,
    best_pgd: float,
    selected_channels: torch.Tensor,
    loader_generator: torch.Generator,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "mep": mep_state.state_dict() if mep_state is not None else None,
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
        "epoch": epoch,
        "best_pgd": best_pgd,
        "selected_channels": selected_channels.detach().cpu(),
        "rng": _rng_state(loader_generator),
    }


def train(config: dict[str, Any], resume: str | None = None) -> Path:
    validate_config(config)
    seed = int(config["seed"])
    seed_everything(seed, bool(config["deterministic"]))
    device = resolve_device(str(config["device"]))
    loaders = build_cifar10_loaders(config["data"], seed)
    model = ResNet18(int(config["model"]["num_classes"])).to(device)

    train_cfg = config["train"]
    epsilon = float(train_cfg["epsilon"]) / 255.0
    alpha = float(train_cfg["alpha"] if train_cfg["alpha"] is not None else train_cfg["epsilon"]) / 255.0
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        momentum=float(train_cfg["momentum"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    iterations_per_epoch = len(loaders.train)
    milestones = [int(value) * iterations_per_epoch for value in train_cfg["milestones"]]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=float(train_cfg["lr_gamma"])
    )

    mep_state: MEPState | None = None
    if train_cfg["backend"] == "mep":
        mep_state = MEPState(loaders.train_size, (3, 32, 32), epsilon, alpha, device)

    run_dir = Path(config["output"]["root"]).expanduser() / str(config["name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {key: value for key, value in config.items() if not key.startswith("_")},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    write_json(run_dir / "environment.json", environment_metadata())

    start_epoch = 0
    best_pgd = -math.inf
    selected_channels = torch.empty(0, dtype=torch.long, device=device)
    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu")
        saved_config = checkpoint.get("config", {})
        saved_train = saved_config.get("train", {})
        for key in (
            "objective",
            "backend",
            "epsilon",
            "alpha",
            "feature_node",
            "fd_include_mep_logit",
        ):
            if key in saved_train and saved_train[key] != train_cfg[key]:
                raise ValueError(
                    f"Resume config mismatch for train.{key}: "
                    f"checkpoint={saved_train[key]!r}, current={train_cfg[key]!r}"
                )
        load_model_state(model, checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if mep_state is not None:
            if checkpoint.get("mep") is None:
                raise ValueError("MEP run cannot resume from a checkpoint without MEP state")
            mep_state.load_state_dict(checkpoint["mep"])
        selected_channels = checkpoint.get("selected_channels", selected_channels).to(device)
        best_pgd = float(checkpoint.get("best_pgd", best_pgd))
        start_epoch = int(checkpoint["epoch"]) + 1
        _restore_rng_state(checkpoint["rng"], loaders.train_generator)
    elif (run_dir / "final.pt").exists():
        raise FileExistsError(f"Run is already complete: {run_dir}")

    objective = str(train_cfg["objective"])
    feature_node = str(train_cfg["feature_node"])
    epochs = int(train_cfg["epochs"])
    for epoch in range(start_epoch, epochs):
        if mep_state is not None and epoch % int(train_cfg["mep_reset_epochs"]) == 0:
            mep_state.reset(epoch)
        model.train()
        started = time.time()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        epoch_scores: torch.Tensor | None = None

        progress = tqdm(loaders.train, desc=f"train {epoch + 1}/{epochs}", leave=False)
        for images, targets, sample_ids in progress:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            clean_features_for_selection = None
            if objective == "induce_co":
                # Eq. (2) ranks channels by clean-vs-adversarial activation
                # differences. This extra forward is confined to the causal
                # induction experiments, not the eight main-table runs.
                with torch.no_grad():
                    _, clean_features_for_selection = _forward_features(model, images)
            if mep_state is not None:
                initial_delta, previous_momentum = mep_state.get(sample_ids, images)
            else:
                initial_delta = random_start(images, epsilon)
                previous_momentum = None
            initial_delta.requires_grad_(True)

            logits_initial, features_initial = _forward_features(model, images + initial_delta)
            initial_loss = smooth_cross_entropy(
                logits_initial, targets, float(train_cfg["label_true_probability"])
            )
            needs_initial_graph = objective in {"mep_baseline", "ours_fd", "induce_co"}
            input_gradient = torch.autograd.grad(
                initial_loss, initial_delta, retain_graph=needs_initial_graph
            )[0].detach()
            perturbation = build_training_perturbation(
                initial=initial_delta,
                inputs=images,
                input_gradient=input_gradient,
                epsilon=epsilon,
                alpha=alpha,
                previous_momentum=previous_momentum,
                momentum_decay=float(train_cfg["mep_momentum_decay"]),
            )
            logits_adv, features_adv = _forward_features(model, images + perturbation.adversarial)
            adversarial_ce = smooth_cross_entropy(
                logits_adv, targets, float(train_cfg["label_true_probability"])
            )

            if objective == "mep_baseline":
                loss = adversarial_ce + float(train_cfg["mep_logit_weight"]) * F.mse_loss(
                    logits_adv, logits_initial
                )
            elif objective == "ours_co":
                loss = adversarial_ce
            elif objective == "ours_fd":
                loss = adversarial_ce + float(train_cfg["feature_weight"]) * feature_difference(
                    features_adv[feature_node], features_initial[feature_node]
                )
                if bool(train_cfg["fd_include_mep_logit"]):
                    loss = loss + float(train_cfg["mep_logit_weight"]) * F.mse_loss(
                        logits_adv, logits_initial
                    )
            elif objective == "induce_co":
                if selected_channels.numel() == 0:
                    loss = adversarial_ce
                else:
                    masked_logits = model(
                        images + perturbation.adversarial,
                        masks={feature_node: selected_channels},
                    )
                    masked_ce = smooth_cross_entropy(
                        masked_logits, targets, float(train_cfg["label_true_probability"])
                    )
                    selected_difference = feature_difference(
                        features_adv[feature_node],
                        features_initial[feature_node],
                        selected_channels,
                    )
                    loss = (
                        adversarial_ce
                        + masked_ce
                        - float(train_cfg["feature_weight"]) * selected_difference
                    )
                assert clean_features_for_selection is not None
                batch_scores = channel_difference_scores(
                    features_adv[feature_node].detach(),
                    clean_features_for_selection[feature_node].detach(),
                )
                weighted = batch_scores * targets.numel()
                epoch_scores = weighted if epoch_scores is None else epoch_scores + weighted
            else:
                raise AssertionError(f"Unhandled objective: {objective}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if mep_state is not None:
                assert perturbation.next_prior is not None and perturbation.next_momentum is not None
                mep_state.update(sample_ids, perturbation.next_prior, perturbation.next_momentum)

            total_loss += loss.item() * targets.numel()
            total_correct += logits_adv.argmax(1).eq(targets).sum().item()
            total_seen += targets.numel()
            progress.set_postfix(loss=f"{total_loss / total_seen:.4f}")

        if objective == "induce_co" and epoch_scores is not None:
            channel_count = epoch_scores.numel()
            count = max(1, math.ceil(channel_count * float(train_cfg["induce_percent"]) / 100.0))
            selected_channels = epoch_scores.topk(count).indices.detach()

        monitor_clean = clean_accuracy(model, loaders.test, device)
        monitor_pgd, epoch_vact = pgd_monitor(
            model,
            loaders.test,
            device,
            epsilon=epsilon,
            steps=int(train_cfg["monitor_pgd_steps"]),
            subset=train_cfg.get("monitor_subset"),
            track_features=bool(train_cfg.get("track_features", False)),
        )
        metrics = {
            "clean_accuracy": monitor_clean,
            "pgd10_accuracy": monitor_pgd,
            "train_loss": total_loss / max(total_seen, 1),
            "train_accuracy": total_correct / max(total_seen, 1),
        }
        row = {
            "epoch": epoch,
            "seconds": time.time() - started,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": metrics["train_loss"],
            "train_accuracy": metrics["train_accuracy"],
            "monitor_clean_accuracy": monitor_clean,
            "monitor_pgd10_accuracy": monitor_pgd,
            "vact_A": "" if epoch_vact is None else epoch_vact["A"],
            "vact_B": "" if epoch_vact is None else epoch_vact["B"],
            "vact_C": "" if epoch_vact is None else epoch_vact["C"],
            "vact_D": "" if epoch_vact is None else epoch_vact["D"],
            "vact_E": "" if epoch_vact is None else epoch_vact["E"],
            "selected_channels": " ".join(map(str, selected_channels.detach().cpu().tolist())),
        }
        append_csv(run_dir / "epochs.csv", row, TRAIN_FIELDS)

        if monitor_pgd >= best_pgd:
            best_pgd = monitor_pgd
            atomic_torch_save(
                _light_checkpoint(model, config, epoch, metrics, selected_channels), run_dir / "best.pt"
            )
        if (epoch + 1) % int(train_cfg["resume_every"]) == 0 or epoch + 1 == epochs:
            atomic_torch_save(
                _resume_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    mep_state=mep_state,
                    config=config,
                    epoch=epoch,
                    best_pgd=best_pgd,
                    selected_channels=selected_channels,
                    loader_generator=loaders.train_generator,
                ),
                run_dir / "resume.pt",
            )

    final_metrics = {
        "clean_accuracy": clean_accuracy(model, loaders.test, device),
        "pgd10_accuracy": pgd_accuracy(
            model,
            loaders.test,
            device,
            epsilon=epsilon,
            steps=int(train_cfg["monitor_pgd_steps"]),
            subset=train_cfg.get("monitor_subset"),
        ),
    }
    atomic_torch_save(
        _light_checkpoint(model, config, epochs - 1, final_metrics, selected_channels),
        run_dir / "final.pt",
    )
    write_json(run_dir / "final_metrics.json", final_metrics)
    return run_dir
