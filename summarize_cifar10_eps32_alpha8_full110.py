#!/usr/bin/env python3
"""Summarize exploratory full-cycle continuations without importing PyTorch."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import yaml


RUN_NAMES = (
    "full_mep_eps32_alpha8_logit10",
    "full_fd_eps32_alpha8_fw50",
    "full_fd_eps32_alpha8_fw200",
)
DEFAULT_ROOT = "/data/cjk/FGSM-MEP-cifar10-eps32-alpha8-full110"
EPOCHS = 110
METRICS = ("train_loss", "train_accuracy", "monitor_clean_accuracy", "monitor_pgd10_accuracy", "lr")
LOSSES = ("train_ce_loss", "train_logit_mse", "train_feature_mse")


def _number(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean in numeric field")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"nonfinite value: {value}")
    return number


def _accuracy(value: object) -> float:
    number = _number(value)
    if not 0 <= number <= 1:
        raise ValueError(f"accuracy outside [0, 1]: {value}")
    return number


def _read_epochs(path: Path, fields: tuple[str, ...]) -> dict[int, dict[str, object]]:
    """Index independently by epoch: CSV order need not match between files."""
    result = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"epoch", *fields}.issubset(reader.fieldnames or []):
            raise ValueError(f"missing columns in {path.name}")
        for row in reader:
            epoch = int(row["epoch"])
            if epoch < 0 or epoch >= EPOCHS or epoch in result:
                raise ValueError(f"duplicate or out-of-range epoch {epoch} in {path.name}")
            parsed = {
                field: (_accuracy if field.endswith("_accuracy") else _number)(row[field])
                for field in fields
            }
            for name, value in row.items():
                if name and name.startswith("vact_") and value not in (None, ""):
                    parsed[name] = _number(value)
            result[epoch] = parsed
    return result


def summarize_run(run: Path) -> dict[str, object]:
    item = dict.fromkeys(("fd", "best_epoch", "best_clean", "best_pgd", "final_clean", "final_pgd", "final_lr", "vact_b"))
    item.update(name=run.name, status="INCOMPLETE", epochs=0, detail="")
    if not run.is_dir():
        item.update(status="MISSING", detail="run directory absent")
        return item
    try:
        if (run / "nonfinite_diagnostic.json").exists():
            raise ValueError("nonfinite_diagnostic.json present")
        required = ("config.yaml", "epochs.csv")
        missing = [name for name in required if not (run / name).is_file()]
        if missing:
            item["detail"] = "missing " + ", ".join(missing)
            return item
        config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
        item["fd"] = _number(config["train"]["feature_weight"])
        epochs = _read_epochs(run / "epochs.csv", METRICS)
        item["epochs"] = len(epochs)
        loss_path = run / "loss_components.csv"
        losses = _read_epochs(loss_path, LOSSES) if loss_path.is_file() else None
        final_path = run / "final_metrics.json"
        final = None
        if final_path.is_file():
            data = json.loads(final_path.read_text(encoding="utf-8"))
            final = {name: _accuracy(data[name]) for name in ("clean_accuracy", "pgd10_accuracy")}
        # Select only after every available row has passed the finite/duplicate checks.
        if epochs:
            best_epoch = max(epochs, key=lambda epoch: (epochs[epoch]["monitor_pgd10_accuracy"], epoch))
            best = epochs[best_epoch]
            item.update(best_epoch=best_epoch, best_clean=best["monitor_clean_accuracy"], best_pgd=best["monitor_pgd10_accuracy"])
        problems = []
        if set(epochs) != set(range(EPOCHS)):
            problems.append("history must contain every epoch 0–109")
        if losses is None:
            problems.append("loss_components.csv absent")
        elif set(losses) != set(epochs):
            problems.append("loss/metric epoch sets differ")
        if final is None:
            problems.append("final_metrics.json absent")
        if problems:
            item["detail"] = "; ".join(problems)
        else:
            item.update(status="COMPLETE", final_clean=final["clean_accuracy"], final_pgd=final["pgd10_accuracy"], final_lr=epochs[109]["lr"], vact_b=epochs[109].get("vact_B"))
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
        # Never present a best model selected from a numerically invalid history.
        item.update(status="INVALID", detail=str(exc), best_epoch=None, best_clean=None, best_pgd=None, final_clean=None, final_pgd=None, final_lr=None, vact_b=None)
    return item


def _format(value: object, *, percent: bool = False) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.2f}" if percent else f"{value:g}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", nargs="?", default=DEFAULT_ROOT)
    root = Path(parser.parse_args(argv).output_root)
    results = [summarize_run(root / name) for name in RUN_NAMES]
    print("# Exploratory CIFAR-10 ε=32/255 full-cycle continuation\n")
    print("Matched-epsilon, no-noise test-set PGD-10 monitor (εE=32/255, step=8/255). "
          "Best checkpoint is selected on that test monitor; this is not a held-out "
          "validation result, full AutoAttack, or the fixed-εE=16/255 noisy Table 2 protocol.\n")
    print("Accuracies are percentages. COMPLETE means finite metric/loss logs for all epochs 0–109 "
          "and valid final metrics, with no nonfinite diagnostic. Partial runs never "
          "reuse a copied 40-epoch final result.\n")
    print("| Run | Status | Epochs | FD weight | Best epoch | Best clean | Best PGD-10 | Final clean | Final PGD-10 | Final LR | Final Vact-B |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for item in results:
        cells = [item["name"], item["status"], f"{item['epochs']}/{EPOCHS}", _format(item["fd"]), _format(item["best_epoch"])]
        cells += [_format(item[name], percent=True) for name in ("best_clean", "best_pgd", "final_clean", "final_pgd")]
        cells += [_format(item["final_lr"]), _format(item["vact_b"])]
        print("| " + " | ".join(cells) + " |")
    details = [item for item in results if item["detail"]]
    if details:
        print()
        for item in details:
            detail = str(item["detail"]).replace("\n", " ")
            print(f"- {item['name']}: {detail}")
    return 0 if all(item["status"] == "COMPLETE" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
