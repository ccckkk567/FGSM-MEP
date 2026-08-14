from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from .attacks import pgd
from .autoattack_adapter import generate_autoattack
from .data import build_cifar10_loaders
from .evaluation import accuracy_on_examples, generate_examples, load_checkpoint_model
from .models.resnet import NODE_CHANNELS
from .utils import resolve_device, seed_everything


def _analysis_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "epsilon": 16.0,
        "step_size": 2.0,
        "steps": 10,
        "restarts": 1,
        "alpha": 100.0,
        "node": "A",
        "thresholds": [1.0, 0.99, 0.5, 0.1],
        "mask_attacks": ["clean", "fgsm", "pgd50", "cw20", "aa"],
        **config.get("analysis", {}),
    }


def activation_statistics(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    epsilon: float,
    step_size: float,
    steps: int,
    restarts: int,
    alpha: float,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    model.eval()
    l2_sums = {node: 0.0 for node in NODE_CHANNELS}
    channel_sums = {
        node: torch.zeros(channels, dtype=torch.float64) for node, channels in NODE_CHANNELS.items()
    }
    total = 0
    spatial_shapes: dict[str, tuple[int, int]] = {}
    for images, targets, _ in tqdm(loader, desc="activation statistics", leave=False):
        images = images.to(device)
        targets = targets.to(device)
        delta = pgd(
            model,
            images,
            targets,
            epsilon,
            step_size,
            steps,
            restarts,
            random_start=True,
        )
        with torch.no_grad():
            _, clean_features = model(images, return_features=True)
            _, adv_features = model(images + delta, return_features=True)
            for node in NODE_CHANNELS:
                difference = adv_features[node] - clean_features[node]
                l2_sums[node] += difference.flatten(1).norm(p=2, dim=1).sum().item()
                channel_sums[node] += difference.square().sum(dim=(0, 2, 3)).double().cpu()
                spatial_shapes[node] = (difference.shape[2], difference.shape[3])
        total += targets.numel()

    vact = {node: value / max(total, 1) for node, value in l2_sums.items()}
    tact: dict[str, torch.Tensor] = {}
    for node, sums in channel_sums.items():
        height, width = spatial_shapes[node]
        raw = sums / (NODE_CHANNELS[node] * height * width)
        tact[node] = torch.tanh(alpha * raw).float()
    return vact, tact


def analyze_features(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    output_dir: str | Path,
) -> Path:
    seed_everything(int(config["seed"]), bool(config["deterministic"]))
    device = resolve_device(str(config["device"]))
    loaders = build_cifar10_loaders(config["data"], int(config["seed"]))
    model, checkpoint = load_checkpoint_model(checkpoint_path, device)
    analysis = _analysis_config(config)
    vact, tact = activation_statistics(
        model,
        loaders.test,
        device,
        epsilon=float(analysis["epsilon"]) / 255.0,
        step_size=float(analysis["step_size"]) / 255.0,
        steps=int(analysis["steps"]),
        restarts=int(analysis["restarts"]),
        alpha=float(analysis["alpha"]),
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "vact.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["node", "vact"])
        writer.writeheader()
        for node, value in vact.items():
            writer.writerow({"node": node, "vact": value})
    with (output / "tact.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["node", "channel", "tact"])
        writer.writeheader()
        for node, values in tact.items():
            for channel, value in enumerate(values.tolist()):
                writer.writerow({"node": node, "channel": channel, "tact": value})

    figure, axes = plt.subplots(1, 5, figsize=(18, 3.2))
    for axis, node in zip(axes, NODE_CHANNELS):
        axis.bar(range(len(tact[node])), tact[node].numpy(), width=0.9)
        axis.set_title(f"Node {node}")
        axis.set_xlabel("Channel")
        axis.set_ylim(0, 1.02)
    axes[0].set_ylabel(r"$T_{act}$")
    figure.suptitle(f"Channel activation differences, checkpoint epoch {checkpoint.get('epoch', '?')}")
    figure.tight_layout()
    figure.savefig(output / "tact.png", dpi=180)
    plt.close(figure)
    return output


def _read_tact(path: Path, node: str) -> torch.Tensor:
    frame = pd.read_csv(path)
    values = frame[frame["node"] == node].sort_values("channel")["tact"].to_numpy()
    return torch.tensor(values, dtype=torch.float32)


def analyze_masks(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    output_dir: str | Path,
) -> Path:
    seed = int(config["seed"])
    seed_everything(seed, bool(config["deterministic"]))
    device = resolve_device(str(config["device"]))
    loaders = build_cifar10_loaders(config["data"], seed)
    model, _ = load_checkpoint_model(checkpoint_path, device)
    analysis = _analysis_config(config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stats_dir = output / "statistics"
    if not (stats_dir / "tact.csv").exists():
        analyze_features(config, checkpoint_path, stats_dir)
    node = str(analysis["node"])
    scores = _read_tact(stats_dir / "tact.csv", node)

    images: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch_images, batch_targets, _ in loaders.test:
        images.append(batch_images)
        targets.append(batch_targets)
    all_images, all_targets = torch.cat(images), torch.cat(targets)
    epsilon = float(analysis["epsilon"]) / 255.0
    step_size = float(analysis["step_size"]) / 255.0
    rows: list[dict[str, Any]] = []
    disabled_noise = {"enabled": False, "kind": "uniform", "magnitude": 0.0, "seed": seed}

    for attack_name in analysis["mask_attacks"]:
        if str(attack_name) in {"aa", "apgd-t"}:
            examples = generate_autoattack(
                model,
                all_images,
                all_targets,
                epsilon=epsilon,
                batch_size=int(config["eval"]["autoattack_batch_size"]),
                seed=seed,
                attack=str(attack_name),
                device=device,
                log_path=output / f"mask_{attack_name}.log",
            )
        else:
            examples = generate_examples(
                model,
                all_images,
                all_targets,
                attack_name=str(attack_name),
                device=device,
                epsilon=epsilon,
                step_size=step_size,
                restarts=int(analysis["restarts"]),
                batch_size=int(config["eval"]["batch_size"]),
            )
        for threshold in analysis["thresholds"]:
            channels = torch.where(scores > float(threshold))[0]
            accuracy = accuracy_on_examples(
                model,
                examples,
                all_targets,
                device=device,
                batch_size=int(config["eval"]["batch_size"]),
                noise_config=disabled_noise,
                noise_seed=seed,
                masks={node: channels.to(device)},
            )
            rows.append(
                {
                    "threshold": float(threshold),
                    "attack": str(attack_name),
                    "masked_channels": channels.numel(),
                    "accuracy": accuracy,
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "mask_results.csv", index=False)
    pivot = frame.pivot(index="threshold", columns="attack", values="accuracy")
    pivot = pivot.reindex([float(value) for value in analysis["thresholds"]])
    axis = (100.0 * pivot).plot(kind="bar", figsize=(8, 4.8))
    axis.set_ylabel("Classification accuracy (%)")
    axis.set_xlabel("Channel masking threshold")
    axis.set_ylim(0, 100)
    axis.figure.tight_layout()
    axis.figure.savefig(output / "mask_results.png", dpi=180)
    plt.close(axis.figure)
    return output


def analyze_induce(run_dirs: list[str | Path], output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4.8))
    combined: list[pd.DataFrame] = []
    for run_value in run_dirs:
        run = Path(run_value)
        config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
        percent = float(config["train"]["induce_percent"])
        frame = pd.read_csv(run / "epochs.csv")
        frame["induce_percent"] = percent
        combined.append(frame)
        axis.plot(frame["epoch"], 100.0 * frame["monitor_clean_accuracy"], "--", label=f"Clean C{percent:g}%")
        axis.plot(frame["epoch"], 100.0 * frame["monitor_pgd10_accuracy"], label=f"PGD10 C{percent:g}%")
    pd.concat(combined, ignore_index=True).to_csv(output / "induce_curves.csv", index=False)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Accuracy (%)")
    axis.set_ylim(0, 100)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "induce_curves.png", dpi=180)
    plt.close(figure)
    return output


def analyze_vact_curves(run_dirs: list[str | Path], output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figure, accuracy_axis = plt.subplots(figsize=(8, 4.8))
    vact_axis = accuracy_axis.twinx()
    combined: list[pd.DataFrame] = []
    colors = {"A": "tab:red", "B": "tab:green", "C": "tab:brown", "D": "tab:purple", "E": "tab:cyan"}
    for run_value in run_dirs:
        run = Path(run_value)
        config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
        label = str(config["train"]["objective"])
        frame = pd.read_csv(run / "epochs.csv")
        if frame["vact_A"].isna().all():
            raise ValueError(f"Run did not enable train.track_features: {run}")
        frame["run"] = label
        combined.append(frame)
        accuracy_axis.plot(
            frame["epoch"],
            frame["monitor_pgd10_accuracy"],
            marker="o" if len(frame) <= 30 else None,
            linewidth=1.5,
            label=f"PGD-10 {label}",
        )
        for node, color in colors.items():
            vact_axis.plot(
                frame["epoch"],
                frame[f"vact_{node}"],
                linestyle="--" if label == "ours_co" else ":",
                color=color,
                alpha=0.75,
                label=f"Node {node} {label}",
            )
    pd.concat(combined, ignore_index=True).to_csv(output / "vact_curves.csv", index=False)
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("PGD-10 accuracy")
    vact_axis.set_ylabel(r"$V_{act}$")
    handles1, labels1 = accuracy_axis.get_legend_handles_labels()
    handles2, labels2 = vact_axis.get_legend_handles_labels()
    accuracy_axis.legend(handles1 + handles2, labels1 + labels2, fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(output / "vact_curves.png", dpi=180)
    plt.close(figure)
    return output
