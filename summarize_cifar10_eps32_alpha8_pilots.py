#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


RUN_NAMES = (
    "pilot_mep_eps32_alpha8_logit10",
    "pilot_fd_eps32_alpha8_fw5",
    "pilot_fd_eps32_alpha8_fw10",
    "pilot_fd_eps32_alpha8_fw25",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize epsilon=32 alpha=8 pilots")
    parser.add_argument(
        "output_root",
        nargs="?",
        default="/data/cjk/FGSM-MEP-cifar10-eps32-alpha8-pilots",
    )
    root = Path(parser.parse_args().output_root)
    results: list[dict[str, object]] = []
    complete = True
    for run_name in RUN_NAMES:
        run = root / run_name
        diagnostic = run / "nonfinite_diagnostic.json"
        if diagnostic.exists():
            details = json.loads(diagnostic.read_text(encoding="utf-8"))
            print(
                f"NONFINITE: {run_name} stage={details['stage']} "
                f"epoch={details['epoch']} batch={details['batch']}"
            )
            complete = False
            continue
        required = (run / "config.yaml", run / "epochs.csv", run / "final_metrics.json")
        if not all(path.exists() for path in required):
            print(f"MISSING: {run_name}")
            complete = False
            continue
        config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
        train = config["train"]
        rows = _read_csv(run / "epochs.csv")
        best_index, best = max(
            enumerate(rows),
            key=lambda item: (float(item[1]["monitor_pgd10_accuracy"]), item[0]),
        )
        losses = _read_csv(run / "loss_components.csv")
        loss = losses[best_index]
        final = json.loads((run / "final_metrics.json").read_text(encoding="utf-8"))
        results.append(
            {
                "name": run_name,
                "objective": train["objective"],
                "weight": float(train["feature_weight"]),
                "epoch": int(best["epoch"]),
                "best_clean": 100 * float(best["monitor_clean_accuracy"]),
                "best_pgd": 100 * float(best["monitor_pgd10_accuracy"]),
                "final_clean": 100 * float(final["clean_accuracy"]),
                "final_pgd": 100 * float(final["pgd10_accuracy"]),
                "ce": float(loss["train_ce_loss"]),
                "logit": 10 * float(loss["train_logit_mse"]),
                "feature": float(train["feature_weight"])
                * float(loss["train_feature_mse"]),
            }
        )

    results.sort(key=lambda item: float(item["best_pgd"]), reverse=True)
    print(
        "| Run | Objective | FD weight | Best epoch | Best clean | Best PGD-10 "
        "| Final clean | Final PGD-10 | CE | 10×logit | λ×feature |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for item in results:
        print(
            f"| {item['name']} | {item['objective']} | {item['weight']:g} "
            f"| {item['epoch']} | {item['best_clean']:.2f} | {item['best_pgd']:.2f} "
            f"| {item['final_clean']:.2f} | {item['final_pgd']:.2f} "
            f"| {item['ce']:.4f} | {item['logit']:.4f} | {item['feature']:.4f} |"
        )
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
