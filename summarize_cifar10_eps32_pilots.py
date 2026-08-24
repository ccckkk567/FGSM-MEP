#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


RUN_NAMES = (
    "pilot_fd_eps32_alpha32_fw25",
    "pilot_fd_eps32_alpha16_fw25",
    "pilot_fd_eps32_alpha16_fw10",
    "pilot_mep_baseline_eps32_alpha16",
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize the CIFAR-10 epsilon=32 pilots")
    parser.add_argument(
        "output_root",
        nargs="?",
        default="/data/cjk/FGSM-MEP-cifar10-fd-pilots",
    )
    args = parser.parse_args()
    root = Path(args.output_root)
    summaries: list[dict[str, object]] = []

    for run_name in RUN_NAMES:
        run = root / run_name
        required = (run / "config.yaml", run / "epochs.csv", run / "final_metrics.json")
        if not all(path.exists() for path in required):
            print(f"MISSING: {run_name}")
            continue

        config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
        train = config["train"]
        epochs = _rows(run / "epochs.csv")
        best_index, best = max(
            enumerate(epochs),
            key=lambda item: (float(item[1]["monitor_pgd10_accuracy"]), item[0]),
        )
        losses_path = run / "loss_components.csv"
        losses = _rows(losses_path) if losses_path.exists() else []
        loss = losses[best_index] if best_index < len(losses) else {}
        final = json.loads((run / "final_metrics.json").read_text(encoding="utf-8"))
        logit_weight = float(train["mep_logit_weight"])
        feature_weight = float(train["feature_weight"])
        logit_term = logit_weight * float(loss.get("train_logit_mse", "nan"))
        feature_term = feature_weight * float(loss.get("train_feature_mse", "nan"))
        summaries.append(
            {
                "name": run_name,
                "objective": train["objective"],
                "alpha": float(train["alpha"]),
                "feature_weight": feature_weight,
                "epoch": int(best["epoch"]),
                "best_clean": 100.0 * float(best["monitor_clean_accuracy"]),
                "best_pgd10": 100.0 * float(best["monitor_pgd10_accuracy"]),
                "final_clean": 100.0 * float(final["clean_accuracy"]),
                "final_pgd10": 100.0 * float(final["pgd10_accuracy"]),
                "ce": float(loss.get("train_ce_loss", "nan")),
                "weighted_logit": logit_term,
                "weighted_feature": feature_term,
            }
        )

    summaries.sort(key=lambda item: float(item["best_pgd10"]), reverse=True)
    print(
        "| Run | Objective | α | FD weight | Best epoch | Best clean | Best PGD-10 "
        "| Final clean | Final PGD-10 | CE | 10×logit MSE | λ×feature MSE |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for item in summaries:
        print(
            f"| {item['name']} | {item['objective']} | {item['alpha']:g}/255 "
            f"| {item['feature_weight']:g} | {item['epoch']} | {item['best_clean']:.2f} "
            f"| {item['best_pgd10']:.2f} | {item['final_clean']:.2f} "
            f"| {item['final_pgd10']:.2f} | {item['ce']:.4f} "
            f"| {item['weighted_logit']:.4f} | {item['weighted_feature']:.4f} |"
        )
    return 0 if len(summaries) == len(RUN_NAMES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
