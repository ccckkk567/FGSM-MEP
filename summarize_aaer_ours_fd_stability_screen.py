#!/usr/bin/env python3
"""Summarize the 40-epoch multi-seed AAER Ours-FD stability screen."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


DEFAULT_ROOT = "/data/cjk/FGSM-MEP-aaer-ours-fd-cifar10-stability-screen"


def _finite(value: object, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"nonfinite {label}: {value}")
    return parsed


def _completed_metrics(run_dir: Path, epochs: int) -> dict[str, float]:
    with (run_dir / "epochs.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if [int(row["epoch"]) for row in rows] != list(range(epochs)):
        raise ValueError(f"epochs.csv must contain exactly 0..{epochs - 1}")
    last = rows[-1]
    final = json.loads((run_dir / "final_metrics.json").read_text(encoding="utf-8"))
    with (run_dir / "loss_components.csv").open(newline="", encoding="utf-8") as handle:
        losses = list(csv.DictReader(handle))
    if len(losses) != epochs:
        raise ValueError("loss_components.csv must contain one row per epoch")
    return {
        "final_clean": _finite(final["clean_accuracy"], "final clean"),
        "final_pgd10": _finite(final["pgd10_accuracy"], "final PGD-10"),
        "final_loss": _finite(last["train_loss"], "final loss"),
        "final_feature_mse": _finite(losses[-1]["train_feature_mse"], "final feature MSE"),
    }


def _status(job: dict[str, Any], epochs: int) -> dict[str, Any]:
    run = Path(str(job["run_dir"]))
    payload: dict[str, Any] = {**job, "status": "MISSING"}
    diagnostic = run / "nonfinite_diagnostic.json"
    if diagnostic.is_file():
        data = json.loads(diagnostic.read_text(encoding="utf-8"))
        return {**payload, "status": "NONFINITE", "stage": data.get("stage", ""), "epoch": data.get("epoch", ""), "batch": data.get("batch", "")}
    try:
        return {**payload, "status": "COMPLETE", **_completed_metrics(run, epochs)}
    except (OSError, KeyError, ValueError, IndexError, json.JSONDecodeError) as error:
        return {**payload, "status": "INVALID", "detail": str(error)}


def _percent(values: list[float]) -> str:
    if not values:
        return "—"
    mean = 100 * statistics.mean(values)
    if len(values) == 1:
        return f"{mean:.2f}"
    return f"{mean:.2f} ± {100 * statistics.stdev(values):.2f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", nargs="?", default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    root = Path(args.output_root)
    manifest = json.loads((root / "screen_manifest.json").read_text(encoding="utf-8"))
    epochs = int(manifest["screen_epochs"])
    results = [_status(job, epochs) for job in manifest["jobs"]]

    grouped: dict[tuple[int, int, float, float], list[dict[str, Any]]] = {}
    for result in results:
        key = (int(result["epsilon"]), int(result["alpha"]), float(result["feature_weight"]), float(result["lr"]))
        grouped.setdefault(key, []).append(result)

    print("# AAER-PreAct Ours-FD multi-seed stability screen\n")
    print(
        "This is a 40-epoch diagnostic screen, not a final Table-2 result. "
        "PGD-10 values use only a 1,000-example monitor; final checkpoint selection is not performed here.\n"
    )
    print("| εT | α | FD weight | lr | Complete seeds | Nonfinite seeds | Final clean | Final monitor PGD-10 | Status |")
    print("|---:|---:|---:|---:|---:|---|---:|---:|---|")
    all_complete = True
    for key in sorted(grouped):
        epsilon, alpha, feature_weight, lr = key
        group = sorted(grouped[key], key=lambda item: int(item["seed"]))
        completed = [item for item in group if item["status"] == "COMPLETE"]
        failed = [item for item in group if item["status"] == "NONFINITE"]
        incomplete = [item for item in group if item["status"] not in {"COMPLETE", "NONFINITE"}]
        if failed:
            status = "NONFINITE"
        elif incomplete:
            status = "INCOMPLETE"
        else:
            status = "ALL_FINITE"
        all_complete &= status == "ALL_FINITE"
        failures = ", ".join(
            f"s{item['seed']}:{item.get('stage', '')}@e{item.get('epoch', '')}/b{item.get('batch', '')}"
            for item in failed
        )
        clean = [float(item["final_clean"]) for item in completed]
        pgd = [float(item["final_pgd10"]) for item in completed]
        print(
            f"| {epsilon}/255 | {alpha}/255 | {feature_weight:g} | {lr:g} | "
            f"{len(completed)}/3 | {failures or '—'} | {_percent(clean)} | {_percent(pgd)} | {status} |"
        )
    return 0 if all_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
