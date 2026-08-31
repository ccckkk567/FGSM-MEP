from __future__ import annotations

from pathlib import Path

import yaml

from .config import apply_overrides, load_config
from .evaluation import evaluate
from .reporting import compare_results, summarize_aaer_table2, summarize_results
from .training import train


def reproduce(
    manifest_path: str | Path,
    *,
    data_root: str | None = None,
    output_root: str | None = None,
    device: str | None = None,
) -> Path:
    manifest_file = Path(manifest_path).resolve()
    manifest = yaml.safe_load(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("experiments"), list):
        raise ValueError("Manifest must contain an experiments list")
    workspace = manifest_file.parents[2] if manifest_file.parent.name == "manifests" else Path.cwd()
    configured_output = output_root or str(manifest.get("output_root", "runs"))
    results: list[Path] = []

    for experiment in manifest["experiments"]:
        if not isinstance(experiment, dict) or "config" not in experiment:
            raise ValueError("Every experiment requires a config")
        config_path = (workspace / str(experiment["config"])).resolve()
        train_config = apply_overrides(
            load_config(config_path), data_root=data_root, output_root=configured_output, device=device
        )
        run_dir = Path(train_config["output"]["root"]) / str(train_config["name"])
        final_path = run_dir / "final.pt"
        if not final_path.exists():
            resume_path = run_dir / "resume.pt"
            train(train_config, resume=str(resume_path) if resume_path.exists() else None)

        checkpoint_name = str(experiment.get("checkpoint", "final.pt"))
        checkpoint_path = run_dir / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Expected checkpoint does not exist: {checkpoint_path}")

        eval_config_path = (workspace / str(experiment.get("eval_config", "configs/eval/paper.yaml"))).resolve()
        eval_config = apply_overrides(
            load_config(eval_config_path), data_root=data_root, output_root=configured_output, device=device
        )
        eval_config["name"] = f"eval_{train_config['name']}"
        result_path = Path(eval_config["output"]["root"]) / eval_config["name"] / "evaluation.json"
        if not result_path.exists():
            result_path = evaluate(eval_config, checkpoint_path)
        results.append(result_path)

    for mechanism_config in manifest.get("mechanism_train_configs", []):
        config = apply_overrides(
            load_config((workspace / str(mechanism_config)).resolve()),
            data_root=data_root,
            output_root=configured_output,
            device=device,
        )
        run_dir = Path(config["output"]["root"]) / str(config["name"])
        if not (run_dir / "final.pt").exists():
            resume_path = run_dir / "resume.pt"
            train(config, resume=str(resume_path) if resume_path.exists() else None)

    report_mode = str(manifest.get("report_mode", "paper")).lower()
    default_report_name = {
        "paper": "paper_table2_report",
        "summary": "sweep_report",
        "aaer": "aaer_table2_report",
    }.get(report_mode, "report")
    report_dir = Path(configured_output) / str(manifest.get("report_name", default_report_name))
    if report_mode == "paper":
        compare_results(results, report_dir)
    elif report_mode == "summary":
        summarize_results(results, report_dir)
    elif report_mode == "aaer":
        summarize_aaer_table2(results, report_dir)
    else:
        raise ValueError("Manifest report_mode must be 'paper', 'summary', or 'aaer'")
    return report_dir
