from __future__ import annotations

import json
from pathlib import Path

import yaml

from co_blessing.config import load_config
from co_blessing.paper_values import PAPER_TABLE2
from co_blessing.reporting import compare_results


ROOT = Path(__file__).resolve().parents[1]


def test_all_training_configs_validate() -> None:
    for path in sorted((ROOT / "configs" / "train").glob("*.yaml")):
        config = load_config(path)
        assert config["data"]["batch_size"] == 128
        assert config["train"]["epochs"] == 110


def test_paper_evaluation_runs_complete_attacks() -> None:
    config = load_config(ROOT / "configs" / "eval" / "paper.yaml")
    assert config["eval"]["freeze_misclassified"] is False
    assert config["eval"]["fgsm_random_start"] is False
    assert config["eval"]["noise"]["clip_to_input_range"] is True


def test_iterative_diagnostic_uses_legacy_reference_attack_semantics() -> None:
    config = load_config(ROOT / "configs" / "eval" / "diagnostic_iterative.yaml")
    assert config["eval"]["freeze_misclassified"] is True
    assert config["eval"]["fgsm_random_start"] is True


def test_mep_fd_configs_include_the_validated_logit_regularizer() -> None:
    paper_fd = load_config(ROOT / "configs" / "train" / "ours_fd_eps12.yaml")
    diagnostic = load_config(
        ROOT / "configs" / "train" / "diagnostic_fd_plus_mep_eps12.yaml"
    )
    assert paper_fd["train"]["fd_include_mep_logit"] is True
    assert diagnostic["train"]["fd_include_mep_logit"] is True

    for epsilon in (10, 12, 14, 16):
        config = load_config(ROOT / "configs" / "train" / f"ours_fd_eps{epsilon}.yaml")
        assert config["train"]["backend"] == "mep"
        assert config["train"]["fd_include_mep_logit"] is True


def test_manifest_paths_exist() -> None:
    manifest = yaml.safe_load(
        (ROOT / "configs" / "manifests" / "cifar10_resnet18.yaml").read_text(encoding="utf-8")
    )
    paths = [entry["config"] for entry in manifest["experiments"]]
    paths += manifest["mechanism_train_configs"]
    assert len(manifest["experiments"]) == 8
    assert all((ROOT / path).exists() for path in paths)


def test_paper_comparison_has_zero_delta_for_exact_values(tmp_path: Path) -> None:
    metrics = {name: value / 100.0 for name, value in PAPER_TABLE2[("ours_fd", 12)].items()}
    result = tmp_path / "evaluation.json"
    result.write_text(
        json.dumps(
            {
                "training_config": {"train": {"objective": "ours_fd", "epsilon": 12}},
                "metrics": metrics,
            }
        ),
        encoding="utf-8",
    )
    csv_path, markdown_path = compare_results([result], tmp_path / "report")
    assert csv_path.exists() and markdown_path.exists()
    assert ",0.0\n" in csv_path.read_text(encoding="utf-8")
