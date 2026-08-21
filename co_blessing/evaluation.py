from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterator

import torch
from tqdm import tqdm

from .attacks import cw_linf, fgsm, parse_attack, pgd
from .autoattack_adapter import generate_autoattack, source_metadata
from .data import build_cifar10_loaders
from .models import ResNet18
from .utils import environment_metadata, load_model_state, resolve_device, seed_everything, write_json


def load_checkpoint_model(
    checkpoint_path: str | Path,
    device: torch.device,
    num_classes: int = 10,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    model = ResNet18(num_classes).to(device)
    load_model_state(model, state)
    model.eval()
    return model, checkpoint


def _noise_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def add_inference_noise(
    inputs: torch.Tensor,
    noise_config: dict[str, Any],
    generator: torch.Generator,
) -> torch.Tensor:
    if not bool(noise_config["enabled"]):
        return inputs
    magnitude = float(noise_config["magnitude"]) / 255.0
    kind = str(noise_config["kind"])
    if kind == "uniform":
        noise = torch.empty_like(inputs).uniform_(-magnitude, magnitude, generator=generator)
    elif kind == "gaussian":
        noise = torch.empty_like(inputs).normal_(0.0, magnitude, generator=generator)
    else:
        raise ValueError(f"Unsupported inference noise: {kind}")
    noisy = inputs + noise
    if bool(noise_config.get("clip_to_input_range", True)):
        noisy = noisy.clamp(0.0, 1.0)
    return noisy


def _attack_seed(base_seed: int, name: str) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return base_seed + int.from_bytes(digest[:2], "big")


def _collect_test(loader: torch.utils.data.DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    images: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch_images, batch_targets, _ in loader:
        images.append(batch_images)
        targets.append(batch_targets)
    return torch.cat(images), torch.cat(targets)


def _batches(
    images: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    for start in range(0, images.shape[0], batch_size):
        end = min(start + batch_size, images.shape[0])
        yield images[start:end], targets[start:end]


def accuracy_on_examples(
    model: torch.nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    noise_config: dict[str, Any],
    noise_seed: int,
    masks: dict[str, torch.Tensor] | None = None,
) -> float:
    generator = _noise_generator(device, noise_seed)
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch_images, batch_targets in _batches(images, targets, batch_size):
            batch_images = batch_images.to(device)
            batch_targets = batch_targets.to(device)
            evaluated = add_inference_noise(batch_images, noise_config, generator)
            logits = model(evaluated, masks=masks)
            correct += logits.argmax(1).eq(batch_targets).sum().item()
            total += batch_targets.numel()
    return correct / max(total, 1)


def generate_examples(
    model: torch.nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    *,
    attack_name: str,
    device: torch.device,
    epsilon: float,
    step_size: float,
    restarts: int,
    batch_size: int,
    freeze_misclassified: bool,
    fgsm_random_start: bool,
) -> torch.Tensor:
    kind, steps = parse_attack(attack_name)
    if kind == "clean":
        return images.clone()
    adversarial: list[torch.Tensor] = []
    for batch_images, batch_targets in tqdm(
        _batches(images, targets, batch_size),
        total=(images.shape[0] + batch_size - 1) // batch_size,
        desc=attack_name,
        leave=False,
    ):
        batch_images = batch_images.to(device)
        batch_targets = batch_targets.to(device)
        if kind == "fgsm":
            delta = fgsm(
                model,
                batch_images,
                batch_targets,
                epsilon,
                random_start=fgsm_random_start,
                freeze_misclassified=freeze_misclassified,
            )
        elif kind == "pgd":
            assert steps is not None
            delta = pgd(
                model,
                batch_images,
                batch_targets,
                epsilon,
                step_size,
                steps,
                restarts,
                random_start=True,
                freeze_misclassified=freeze_misclassified,
            )
        elif kind == "cw":
            assert steps is not None
            delta = cw_linf(
                model,
                batch_images,
                batch_targets,
                epsilon,
                step_size,
                steps,
                restarts,
                freeze_misclassified=freeze_misclassified,
            )
        else:
            raise ValueError(f"Use the AutoAttack adapter for {kind}")
        adversarial.append((batch_images + delta).detach().cpu())
    return torch.cat(adversarial)


def evaluate(config: dict[str, Any], checkpoint_path: str | Path) -> Path:
    seed = int(config["seed"])
    seed_everything(seed, bool(config["deterministic"]))
    device = resolve_device(str(config["device"]))
    loaders = build_cifar10_loaders(config["data"], seed)
    model, checkpoint = load_checkpoint_model(
        checkpoint_path, device, int(config["model"]["num_classes"])
    )
    images, targets = _collect_test(loaders.test)
    eval_cfg = config["eval"]
    epsilon = float(eval_cfg["epsilon"]) / 255.0
    step_size = float(eval_cfg["step_size"]) / 255.0
    batch_size = int(eval_cfg["batch_size"])
    noise_cfg = eval_cfg["noise"]
    base_noise_seed = int(noise_cfg["seed"])

    output_dir = Path(config["output"]["root"]).expanduser() / str(config["name"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, float] = {}
    for attack_name in eval_cfg["attacks"]:
        normalized, _ = parse_attack(str(attack_name))
        attack_seed = _attack_seed(seed, str(attack_name))
        seed_everything(attack_seed, bool(config["deterministic"]))
        if normalized in {"apgd-t", "aa"}:
            examples = generate_autoattack(
                model,
                images,
                targets,
                epsilon=epsilon,
                batch_size=int(eval_cfg["autoattack_batch_size"]),
                seed=attack_seed,
                attack=normalized,
                device=device,
                log_path=output_dir / f"{normalized}.log",
            )
        else:
            examples = generate_examples(
                model,
                images,
                targets,
                attack_name=str(attack_name),
                device=device,
                epsilon=epsilon,
                step_size=step_size,
                restarts=int(eval_cfg["restarts"]),
                batch_size=batch_size,
                freeze_misclassified=bool(eval_cfg["freeze_misclassified"]),
                fgsm_random_start=bool(eval_cfg["fgsm_random_start"]),
            )
        metrics[str(attack_name)] = accuracy_on_examples(
            model,
            examples,
            targets,
            device=device,
            batch_size=batch_size,
            noise_config=noise_cfg,
            noise_seed=base_noise_seed + attack_seed,
        )

    payload = {
        "protocol": "paper_non_adaptive_attack_then_noise",
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "training_config": checkpoint.get("config") if isinstance(checkpoint, dict) else None,
        "evaluation_config": eval_cfg,
        "metrics": metrics,
        "autoattack": source_metadata(),
        "environment": environment_metadata(),
    }
    result_path = output_dir / "evaluation.json"
    write_json(result_path, payload)
    return result_path
