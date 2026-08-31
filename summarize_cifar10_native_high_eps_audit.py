#!/usr/bin/env python3
"""Summarize fixed-configuration high-epsilon failure audits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


EPSILONS = (32, 48, 64)
DEFAULT_ROOT = "/data/cjk/FGSM-MEP-cifar10-native-high-eps-audit"


def _valid_native_config(path: Path, epsilon: int) -> tuple[bool, str]:
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        train = config["train"]
        checks = {
            "objective": "ours_fd", "backend": "mep", "epsilon": epsilon,
            "alpha": epsilon, "lr": 0.1, "mep_logit_weight": 10,
            "fd_include_mep_logit": True, "feature_node": "B", "feature_weight": 200,
            "epochs": 1, "abort_on_nonfinite": True,
        }
        bad = [key for key, expected in checks.items() if train.get(key) != expected]
        return (not bad, ", ".join(bad))
    except (OSError, TypeError, KeyError, yaml.YAMLError) as error:
        return False, str(error)


def _details(payload: dict[str, Any]) -> str:
    tensors = payload.get("tensors", {})
    if not isinstance(tensors, dict):
        return ""
    return ", ".join(
        name for name, stats in tensors.items()
        if isinstance(stats, dict) and float(stats.get("finite_fraction", 1.0)) < 1.0
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", nargs="?", default=DEFAULT_ROOT)
    root = Path(parser.parse_args(argv).output_root)
    complete = True
    print("# Native Ours-FD high-epsilon failure audit\n")
    print("All rows use the frozen baseline prefix: α=εT, lr=0.1, MEP + 10×logit MSE, "
          "FD(B)=200. A finite one-epoch prefix is not a successful baseline; it must "
          "continue unchanged to 110 epochs.\n")
    print("| εT | Status | Stage | Epoch | Batch | Nonfinite tensors | Next action |")
    print("|---:|---|---|---:|---:|---|---|")
    for epsilon in EPSILONS:
        run = root / f"native_audit_ours_fd_eps{epsilon}"
        config_ok, detail = _valid_native_config(run / "config.yaml", epsilon)
        diagnostic = run / "nonfinite_diagnostic.json"
        final = run / "final.pt"
        if not run.exists():
            complete = False
            row = ("MISSING", "", "", "", "", "run audit")
        elif not config_ok:
            complete = False
            row = ("INVALID_CONFIG", "", "", "", detail, "fix configuration")
        elif diagnostic.exists():
            payload = json.loads(diagnostic.read_text(encoding="utf-8"))
            row = (
                "NUMERICAL_DIVERGENCE", str(payload.get("stage", "")),
                str(payload.get("epoch", "")), str(payload.get("batch", "")),
                _details(payload), "record as N/A; do not run attacks",
            )
        elif final.exists():
            row = ("FINITE_PREFIX", "completed", "0", "", "", "continue unchanged to 110 epochs")
        else:
            complete = False
            row = ("INCOMPLETE", "", "", "", "", "inspect log")
        print(f"| {epsilon}/255 | " + " | ".join(row) + " |")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
