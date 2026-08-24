from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "name": "experiment",
    "seed": 0,
    "deterministic": False,
    "device": "auto",
    "data": {
        "dataset": "cifar10",
        "root": "data",
        "download": True,
        "batch_size": 128,
        "num_workers": 0,
        "train_subset": None,
        "test_subset": None,
    },
    "model": {"name": "resnet18", "num_classes": 10},
    "train": {
        "objective": "ours_fd",
        "backend": "mep",
        "epochs": 110,
        "epsilon": 12.0,
        "alpha": None,
        "lr": 0.1,
        "milestones": [100, 105],
        "lr_gamma": 0.1,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "label_true_probability": 0.6,
        "mep_reset_epochs": 40,
        "mep_momentum_decay": 0.3,
        "mep_logit_weight": 10.0,
        "fd_include_mep_logit": True,
        "feature_node": "B",
        "feature_weight": 200.0,
        "induce_percent": 10.0,
        "resume_every": 1,
        "monitor_pgd_steps": 10,
        "monitor_pgd_step_size": 2.0,
        "monitor_subset": None,
        "track_features": False,
        "abort_on_nonfinite": False,
    },
    "eval": {
        "epsilon": 16.0,
        "step_size": 2.0,
        "attacks": ["clean", "fgsm", "pgd10", "pgd20", "pgd50", "cw20", "apgd-t", "aa"],
        "restarts": 1,
        "freeze_misclassified": False,
        "fgsm_random_start": False,
        "noise": {
            "enabled": True,
            "kind": "uniform",
            "magnitude": 16.0,
            "seed": 0,
            "clip_to_input_range": True,
        },
        "batch_size": 128,
        "autoattack_batch_size": 250,
    },
    "output": {"root": "runs"},
}


VALID_OBJECTIVES = {"mep_baseline", "ours_co", "ours_fd", "induce_co"}
VALID_BACKENDS = {"mep", "rs"}
VALID_NODES = {"A", "B", "C", "D", "E"}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must contain a mapping: {config_path}")
    config = _deep_update(copy.deepcopy(DEFAULT_CONFIG), raw)
    config["_config_path"] = str(config_path)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    train = config["train"]
    if str(config["data"].get("dataset", "")).lower() != "cifar10":
        raise ValueError("The first reproduction stage supports only CIFAR-10")
    if str(config["model"].get("name", "")).lower() != "resnet18":
        raise ValueError("The first reproduction stage supports only ResNet18")
    if int(config["model"].get("num_classes", 0)) != 10:
        raise ValueError("CIFAR-10 ResNet18 must use 10 classes")
    if train["objective"] not in VALID_OBJECTIVES:
        raise ValueError(f"Unsupported training objective: {train['objective']}")
    if train["backend"] not in VALID_BACKENDS:
        raise ValueError(f"Unsupported training backend: {train['backend']}")
    if train["feature_node"] not in VALID_NODES:
        raise ValueError(f"Unsupported feature node: {train['feature_node']}")
    if float(train["epsilon"]) <= 0:
        raise ValueError("Training epsilon must be positive")
    if train["alpha"] is not None and float(train["alpha"]) <= 0:
        raise ValueError("Training alpha must be positive")
    if int(config["data"]["batch_size"]) <= 0:
        raise ValueError("Batch size must be positive")
    if int(train["epochs"]) <= 0:
        raise ValueError("Epoch count must be positive")
    if int(train["mep_reset_epochs"]) <= 0:
        raise ValueError("MEP reset period must be positive")
    if float(train["monitor_pgd_step_size"]) <= 0:
        raise ValueError("Monitor PGD step size must be positive")
    if train["objective"] == "induce_co":
        percent = float(train["induce_percent"])
        if not 0 < percent <= 100:
            raise ValueError("induce_percent must be in (0, 100]")
    evaluation = config["eval"]
    if float(evaluation["epsilon"]) <= 0 or float(evaluation["step_size"]) <= 0:
        raise ValueError("Evaluation epsilon and step size must be positive")
    for attack in evaluation["attacks"]:
        value = str(attack).lower()
        if value not in {"clean", "fgsm", "apgd-t", "aa"}:
            prefix_length = 2 if value.startswith("cw") else 3
            if not (
                (value.startswith("pgd") or value.startswith("cw"))
                and value[prefix_length:].isdigit()
            ):
                raise ValueError(f"Invalid evaluation attack: {attack}")
    if evaluation["noise"]["kind"] not in {"uniform", "gaussian"}:
        raise ValueError("Inference noise kind must be uniform or gaussian")
    analysis = config.get("analysis")
    if analysis is not None and analysis.get("node", "A") not in VALID_NODES:
        raise ValueError(f"Unsupported analysis node: {analysis.get('node')}")


def apply_overrides(
    config: dict[str, Any],
    *,
    data_root: str | None = None,
    output_root: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    config = copy.deepcopy(config)
    if data_root is not None:
        config["data"]["root"] = data_root
    if output_root is not None:
        config["output"]["root"] = output_root
    if device is not None:
        config["device"] = device
    validate_config(config)
    return config
