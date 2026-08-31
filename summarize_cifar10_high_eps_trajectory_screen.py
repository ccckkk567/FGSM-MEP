#!/usr/bin/env python3
"""Summarize finite 40-epoch high-epsilon Ours-FD CO-trajectory screens."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_ROOT = "/data/cjk/FGSM-MEP-cifar10-high-eps-trajectory-screen"


def _finite(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite {label}: {value}")
    return result


def _read_epochs(path: Path, expected_epochs: int) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"epoch", "train_loss", "monitor_clean_accuracy", "monitor_pgd10_accuracy", "vact_B"}
    if not rows or not required <= set(rows[0]):
        raise ValueError("epochs.csv is missing required trajectory fields")
    if [int(row["epoch"]) for row in rows] != list(range(expected_epochs)):
        raise ValueError(f"epochs.csv must contain exactly epochs 0..{expected_epochs - 1}")
    parsed = []
    for row in rows:
        item = {name: _finite(row[name], name) for name in required - {"epoch"}}
        if not all(0 <= item[name] <= 1 for name in ("monitor_clean_accuracy", "monitor_pgd10_accuracy")):
            raise ValueError("accuracy outside [0, 1]")
        parsed.append(item)
    return parsed


def summarize_job(job: dict[str, Any], *, screen_epochs: int) -> dict[str, object]:
    result: dict[str, object] = {**job, "status": "MISSING", "detail": ""}
    run = Path(str(job["run_dir"]))
    diagnostic = run / "nonfinite_diagnostic.json"
    if diagnostic.is_file():
        payload = json.loads(diagnostic.read_text(encoding="utf-8"))
        result.update(status="NONFINITE", stage=payload.get("stage", ""), batch=payload.get("batch", ""))
        return result
    try:
        history = _read_epochs(run / "epochs.csv", screen_epochs)
        final = json.loads((run / "final_metrics.json").read_text(encoding="utf-8"))
        final_clean = _finite(final["clean_accuracy"], "final clean")
        final_pgd = _finite(final["pgd10_accuracy"], "final PGD-10")
        if not 0 <= final_clean <= 1 or not 0 <= final_pgd <= 1:
            raise ValueError("final accuracy outside [0, 1]")
        best_epoch = max(range(screen_epochs), key=lambda epoch: history[epoch]["monitor_pgd10_accuracy"])
        best = history[best_epoch]
        result.update(
            status="COMPLETE", stage="completed", batch="", best_epoch=best_epoch,
            best_clean=best["monitor_clean_accuracy"], best_pgd=best["monitor_pgd10_accuracy"],
            final_clean=final_clean, final_pgd=final_pgd,
            pgd_drop=best["monitor_pgd10_accuracy"] - final_pgd,
            final_vact_b=history[-1]["vact_B"],
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        result.update(status="INVALID", detail=str(exc))
    return result


def _percent(value: object) -> str:
    return "—" if value is None else f"{100 * float(value):.2f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", nargs="?", default=DEFAULT_ROOT)
    root = Path(parser.parse_args(argv).output_root)
    payload = json.loads((root / "trajectory_manifest.json").read_text(encoding="utf-8"))
    epochs = int(payload["screen_epochs"])
    results = [summarize_job(job, screen_epochs=epochs) for job in payload["jobs"]]

    print("# CIFAR-10 high-epsilon CO trajectory screen\n")
    print("This is a tuned diagnostic track, not the frozen formal Ours-FD baseline. "
          "‘Best’ is only the maximum matched-epsilon test PGD-10 observed during the "
          f"{epochs}-epoch screen; it is not a model-selection result. A large best-to-final "
          "PGD-10 drop is evidence to inspect for CO, not by itself a pass/fail criterion.\n")
    print("| εT | α | lr | Rationale | Status | Best epoch | Best clean | Best PGD-10 | Final clean | Final PGD-10 | Best→final PGD drop | Final Vact-B |")
    print("|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for item in results:
        if item["status"] == "NONFINITE":
            status = f"NONFINITE ({item.get('stage', '')}, batch {item.get('batch', '')})"
        else:
            status = str(item["status"])
        cells = [
            f"{item['epsilon']}/255", f"{item['alpha']}/255", f"{float(item['lr']):g}",
            str(item["rationale"]), status,
            "—" if item.get("best_epoch") is None else str(item["best_epoch"]),
            *[_percent(item.get(key)) for key in ("best_clean", "best_pgd", "final_clean", "final_pgd", "pgd_drop")],
            "—" if item.get("final_vact_b") is None else f"{float(item['final_vact_b']):.5f}",
        ]
        print("| " + " | ".join(cells) + " |")
    details = [item for item in results if item.get("detail")]
    if details:
        print()
        for item in details:
            print(f"- {item['name']}: {str(item['detail']).replace(chr(10), ' ')}")
    return 0 if all(item["status"] == "COMPLETE" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
