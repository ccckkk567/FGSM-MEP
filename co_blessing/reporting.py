from __future__ import annotations

import csv
import json
import statistics
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
AAER_PROTOCOL = "aaer_official_pgd50_10"
AAER_EPSILONS = (8, 12, 16, 32, 48, 64)
AAER_SEEDS = (0, 1, 2)


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


def summarize_aaer_table2(
    result_paths: list[str | Path],
    output_dir: str | Path,
    *,
    epsilons: tuple[int, ...] = AAER_EPSILONS,
    seeds: tuple[int, ...] = AAER_SEEDS,
) -> tuple[Path, Path]:
    """Aggregate final-checkpoint Ours-FD runs into an AAER-style table.

    This deliberately validates provenance instead of silently averaging a
    mixture of legacy Table-2 runs, best checkpoints, or noisy evaluations.
    Missing runs are represented explicitly so a partially finished sweep
    cannot be mistaken for a complete three-seed result.
    """

    groups: dict[tuple[str, str, str, int], dict[int, dict[str, float]]] = {}
    for result_path in result_paths:
        payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
        if payload.get("protocol") != AAER_PROTOCOL:
            raise ValueError(f"{result_path}: not an AAER Table-2 evaluation")
        if payload.get("checkpoint_role") != "final":
            raise ValueError(f"{result_path}: AAER summary requires final.pt")
        evaluation = payload.get("evaluation_config") or {}
        if bool((evaluation.get("noise") or {}).get("enabled")):
            raise ValueError(f"{result_path}: AAER evaluation must not add inference noise")
        if evaluation.get("attacks") != ["clean", "pgd50"]:
            raise ValueError(f"{result_path}: AAER evaluation requires exactly clean and pgd50")
        if int(evaluation.get("restarts", 0)) != 10:
            raise ValueError(f"{result_path}: AAER evaluation requires 10 PGD restarts")
        if not bool(evaluation.get("freeze_misclassified")):
            raise ValueError(f"{result_path}: AAER official PGD must freeze misclassified samples")
        epsilon_eval = int(round(float(evaluation["epsilon"])))
        if abs(float(evaluation.get("step_size", 0.0)) - epsilon_eval / 4.0) > 1e-12:
            raise ValueError(f"{result_path}: AAER PGD step size must equal epsilon / 4")
        metrics = payload.get("metrics") or {}
        if not {"clean", "pgd50"}.issubset(metrics):
            raise ValueError(f"{result_path}: requires clean and pgd50 metrics")
        training = payload.get("training_config") or {}
        data = training.get("data") or {}
        model = training.get("model") or {}
        train = training.get("train") or {}
        epsilon_train = int(round(float(train["epsilon"])))
        if epsilon_train != epsilon_eval:
            raise ValueError(
                f"{result_path}: matched-epsilon protocol violated "
                f"({epsilon_train}/255 train vs {epsilon_eval}/255 eval)"
            )
        seed = int(training["seed"])
        key = (
            str(data.get("dataset", "")),
            str(model.get("name", "")),
            str(train.get("objective", "")),
            epsilon_train,
        )
        if seed in groups.setdefault(key, {}):
            raise ValueError(f"Duplicate AAER result for {key}, seed={seed}")
        groups[key][seed] = {metric: float(metrics[metric]) * 100.0 for metric in ("clean", "pgd50")}

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "aaer_table2_summary.csv"
    markdown_path = output / "aaer_table2_summary.md"
    fieldnames = [
        "dataset", "model", "method", "epsilon", "metric", "n", "seeds",
        "mean", "std", "status",
    ]
    rows: list[dict[str, Any]] = []
    for dataset, model, method in sorted({key[:3] for key in groups}):
        for epsilon in epsilons:
            seed_values = groups.get((dataset, model, method, epsilon), {})
            for metric in ("clean", "pgd50"):
                values = [seed_values[seed][metric] for seed in seeds if seed in seed_values]
                missing = [seed for seed in seeds if seed not in seed_values]
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "method": method,
                        "epsilon": epsilon,
                        "metric": metric,
                        "n": len(values),
                        "seeds": ",".join(map(str, sorted(seed_values))),
                        "mean": statistics.mean(values) if values else "",
                        "std": statistics.stdev(values) if len(values) > 1 else "",
                        "status": "complete" if not missing else f"missing seeds: {','.join(map(str, missing))}",
                    }
                )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# AAER-compatible Ours-FD Table 2 summary",
        "",
        "Final checkpoints only. Values are percentages; PGD mirrors AAER's released evaluator: matched epsilon, 50 steps, epsilon/4 step size, 10 restarts, and correctness-based sample freezing.",
        "",
        "| Dataset | Model | Method / metric | " + " | ".join(f"{epsilon}/255" for epsilon in epsilons) + " |",
        "|---|---|---|" + "---:|" * len(epsilons),
    ]
    group_names = sorted({key[:3] for key in groups})
    for dataset, model, method in group_names:
        for metric in ("clean", "pgd50"):
            values: list[str] = []
            for epsilon in epsilons:
                row = next(
                    item for item in rows
                    if item["dataset"] == dataset and item["model"] == model
                    and item["method"] == method and item["epsilon"] == epsilon
                    and item["metric"] == metric
                )
                if row["status"] == "complete":
                    values.append(f"{float(row['mean']):.2f} ± {float(row['std']):.2f}")
                else:
                    values.append(f"MISSING ({row['n']}/3)")
            lines.append(f"| {dataset} | {model} | {method} {metric} | " + " | ".join(values) + " |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, markdown_path
