from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .paper_values import PAPER_TABLE2, PAPER_TABLE3


FIELDS = ["table", "method", "epsilon_train", "metric", "paper", "reproduced", "delta"]
SWEEP_FIELDS = [
    "dataset",
    "model",
    "method",
    "seed",
    "epsilon_train",
    "epsilon_eval",
    "checkpoint_epoch",
    "metric",
    "accuracy",
]
METRIC_ORDER = ["clean", "fgsm", "pgd10", "pgd20", "pgd50", "cw20", "apgd-t", "aa"]


def _identity(payload: dict[str, Any]) -> tuple[tuple[str, int], dict[str, float], str]:
    training = payload.get("training_config") or {}
    train = training.get("train") or {}
    method = str(train.get("objective", ""))
    epsilon = int(round(float(train.get("epsilon"))))
    key = (method, epsilon)
    evaluation = payload.get("evaluation_config") or {}
    noise = evaluation.get("noise") or {}
    if noise.get("enabled") is False:
        if key not in PAPER_TABLE3:
            raise ValueError(f"No no-noise Table 3 target for {key}")
        return key, PAPER_TABLE3[key], "Table 3"
    if key not in PAPER_TABLE2:
        raise ValueError(f"No noisy Table 2 target for {key}")
    return key, PAPER_TABLE2[key], "Table 2"


def compare_results(result_paths: list[str | Path], output_dir: str | Path) -> tuple[Path, Path]:
    rows: list[dict[str, Any]] = []
    for result_path in result_paths:
        payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
        (method, epsilon), targets, table = _identity(payload)
        for metric, paper_value in targets.items():
            reproduced = float(payload["metrics"][metric]) * 100.0
            rows.append(
                {
                    "table": table,
                    "method": method,
                    "epsilon_train": epsilon,
                    "metric": metric,
                    "paper": paper_value,
                    "reproduced": reproduced,
                    "delta": reproduced - paper_value,
                }
            )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "paper_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = output / "paper_comparison.md"
    lines = [
        "# Paper result comparison",
        "",
        "All values are percentages. Delta is reproduced minus paper; no hard pass threshold is applied.",
        "",
        "| Table | Method | εT | Metric | Paper | Reproduced | Delta |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['table']} | {row['method']} | {row['epsilon_train']}/255 | {row['metric']} | "
            f"{row['paper']:.2f} | {row['reproduced']:.2f} | {row['delta']:+.2f} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, markdown_path


def summarize_results(result_paths: list[str | Path], output_dir: str | Path) -> tuple[Path, Path]:
    """Write an arbitrary epsilon sweep without requiring a matching paper table row."""
    runs: list[dict[str, Any]] = []
    metric_names: set[str] = set()
    long_rows: list[dict[str, Any]] = []

    for result_path in result_paths:
        payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
        training = payload.get("training_config") or {}
        train = training.get("train") or {}
        data = training.get("data") or {}
        model = training.get("model") or {}
        evaluation = payload.get("evaluation_config") or {}
        metrics = {str(name): float(value) * 100.0 for name, value in payload["metrics"].items()}
        metric_names.update(metrics)
        identity = {
            "dataset": str(data.get("dataset", "")),
            "model": str(model.get("name", "")),
            "method": str(train.get("objective", "")),
            "seed": str(training.get("seed", "")),
            "epsilon_train": float(train["epsilon"]),
            "epsilon_eval": float(evaluation["epsilon"]),
            "checkpoint_epoch": payload.get("checkpoint_epoch"),
        }
        runs.append({**identity, "metrics": metrics})
        for metric, accuracy in metrics.items():
            long_rows.append({**identity, "metric": metric, "accuracy": accuracy})

    runs.sort(
        key=lambda row: (row["dataset"], row["method"], row["seed"], row["epsilon_train"])
    )
    long_rows.sort(
        key=lambda row: (
            row["dataset"],
            row["method"],
            row["seed"],
            row["epsilon_train"],
            row["metric"],
        )
    )
    ordered_metrics = [name for name in METRIC_ORDER if name in metric_names]
    ordered_metrics += sorted(metric_names.difference(ordered_metrics))

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "sweep_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SWEEP_FIELDS)
        writer.writeheader()
        writer.writerows(long_rows)

    markdown_path = output / "sweep_summary.md"
    header = ["Dataset", "Model", "Method", "Seed", "εT", "εE", "Checkpoint epoch", *ordered_metrics]
    lines = [
        "# Epsilon sweep summary",
        "",
        "All accuracies are percentages. Checkpoint selection and noise settings follow the manifest.",
        "",
        "| " + " | ".join(header) + " |",
        "| "
        + " | ".join(
            ["---", "---", "---", "---:", "---:", "---:", "---:"]
            + ["---:"] * len(ordered_metrics)
        )
        + " |",
    ]
    for run in runs:
        values = [
            run["dataset"],
            run["model"],
            run["method"],
            str(run["seed"]),
            f"{run['epsilon_train']:g}/255",
            f"{run['epsilon_eval']:g}/255",
            str(run["checkpoint_epoch"]),
        ]
        values += [f"{run['metrics'][name]:.2f}" if name in run["metrics"] else "" for name in ordered_metrics]
        lines.append("| " + " | ".join(values) + " |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, markdown_path
