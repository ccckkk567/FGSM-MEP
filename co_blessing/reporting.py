from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .paper_values import PAPER_TABLE2, PAPER_TABLE3


FIELDS = ["table", "method", "epsilon_train", "metric", "paper", "reproduced", "delta"]


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
