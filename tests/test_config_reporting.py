from __future__ import annotations

import json
from pathlib import Path

import yaml

from co_blessing.config import load_config
from co_blessing.paper_values import PAPER_TABLE2
from co_blessing.reporting import compare_results, summarize_aaer_table2, summarize_results


ROOT = Path(__file__).resolve().parents[1]


def test_all_training_configs_validate() -> None:
    for path in sorted((ROOT / "configs" / "train").glob("*.yaml")):
        config = load_config(path)
        assert config["data"]["batch_size"] == 128
        assert config["train"]["epochs"] > 0


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

    for epsilon in (8, 10, 12, 14, 16, 32, 48, 64):
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


def test_cifar10_fd_sweep_manifests_are_complete_and_disjoint() -> None:
    manifest_dir = ROOT / "configs" / "manifests"
    full = yaml.safe_load(
        (manifest_dir / "cifar10_fd_sweep.yaml").read_text(encoding="utf-8")
    )
    gpu0 = yaml.safe_load(
        (manifest_dir / "cifar10_fd_sweep_gpu0.yaml").read_text(encoding="utf-8")
    )
    gpu1 = yaml.safe_load(
        (manifest_dir / "cifar10_fd_sweep_gpu1.yaml").read_text(encoding="utf-8")
    )

    def epsilon_set(manifest: dict) -> set[int]:
        values = set()
        for experiment in manifest["experiments"]:
            assert (ROOT / experiment["config"]).exists()
            assert (ROOT / experiment["eval_config"]).exists()
            values.add(int(load_config(ROOT / experiment["config"])["train"]["epsilon"]))
        return values

    assert full["report_mode"] == "summary"
    assert epsilon_set(full) == {8, 12, 16, 32, 48, 64}
    assert epsilon_set(gpu0).isdisjoint(epsilon_set(gpu1))
    assert epsilon_set(gpu0) | epsilon_set(gpu1) == epsilon_set(full)


def test_aaer_ours_fd_manifest_is_final_checkpoint_preact_three_seed_matrix() -> None:
    manifest = yaml.safe_load(
        (ROOT / "configs" / "manifests" / "aaer_ours_fd_cifar10.yaml").read_text(encoding="utf-8")
    )
    assert manifest["report_mode"] == "aaer"
    assert len(manifest["experiments"]) == 18
    observed: set[tuple[int, int]] = set()
    for item in manifest["experiments"]:
        assert item["checkpoint"] == "final.pt"
        config = load_config(ROOT / item["config"])
        evaluation = load_config(ROOT / item["eval_config"])
        assert config["data"]["dataset"] == "cifar10"
        assert config["model"]["name"] == "preactresnet18"
        assert config["model"]["input_normalization"] == "cifar10"
        assert config["train"]["epochs"] == 110
        assert config["train"]["mep_reset_epochs"] == 40
        assert config["train"]["milestones"] == [100, 105]
        assert config["train"]["fd_include_mep_logit"] is True
        assert evaluation["eval"]["protocol"] == "aaer_official_pgd50_10"
        assert evaluation["eval"]["attacks"] == ["clean", "pgd50"]
        assert evaluation["eval"]["restarts"] == 10
        assert evaluation["eval"]["freeze_misclassified"] is True
        assert evaluation["eval"]["step_size"] == evaluation["eval"]["epsilon"] / 4
        observed.add((int(config["train"]["epsilon"]), int(config["seed"])))
    assert observed == {(epsilon, seed) for epsilon in (8, 12, 16, 32, 48, 64) for seed in range(3)}


def test_eps32_pilot_grid() -> None:
    expected = {
        "pilot_fd_eps32_alpha32_fw25.yaml": ("ours_fd", 32.0, 25.0),
        "pilot_fd_eps32_alpha16_fw25.yaml": ("ours_fd", 16.0, 25.0),
        "pilot_fd_eps32_alpha16_fw10.yaml": ("ours_fd", 16.0, 10.0),
        "pilot_mep_baseline_eps32_alpha16.yaml": ("mep_baseline", 16.0, 0.0),
    }
    for name, (objective, alpha, feature_weight) in expected.items():
        config = load_config(ROOT / "configs" / "train" / name)
        train = config["train"]
        assert config["deterministic"] is True
        assert train["backend"] == "mep"
        assert train["objective"] == objective
        assert train["epochs"] == 40
        assert train["epsilon"] == 32
        assert train["alpha"] == alpha
        assert train["feature_weight"] == feature_weight
        assert train["monitor_pgd_step_size"] == 8


def test_eps32_nonfinite_diagnostic_grid() -> None:
    names = (
        "diagnostic_mep_eps32_alpha16_logit10_lr01.yaml",
        "diagnostic_mep_eps32_alpha8_logit10_lr01.yaml",
        "diagnostic_mep_ce_eps32_alpha8_lr01.yaml",
        "diagnostic_mep_eps32_alpha8_logit10_lr001.yaml",
    )
    for name in names:
        config = load_config(ROOT / "configs" / "train" / name)
        train = config["train"]
        assert config["deterministic"] is True
        assert train["backend"] == "mep"
        assert train["epochs"] == 1
        assert train["epsilon"] == 32
        assert train["abort_on_nonfinite"] is True


def test_native_high_epsilon_failure_audits_are_frozen_ours_fd_prefixes() -> None:
    for epsilon in (32, 48, 64):
        config = load_config(ROOT / "configs" / "train" / f"native_audit_ours_fd_eps{epsilon}.yaml")
        baseline = load_config(ROOT / "configs" / "train" / f"ours_fd_eps{epsilon}.yaml")
        train = config["train"]
        baseline_train = baseline["train"].copy()
        baseline_train.update({"epochs": 1, "alpha": epsilon, "abort_on_nonfinite": True})
        assert config["seed"] == 0
        assert config["data"] == baseline["data"]
        assert config["model"] == baseline["model"]
        assert train == baseline_train
        assert train["objective"] == "ours_fd"
        assert train["backend"] == "mep"
        assert train["epochs"] == 1
        assert train["epsilon"] == epsilon
        assert train["alpha"] == epsilon
        assert train["lr"] == 0.1
        assert train["mep_logit_weight"] == 10
        assert train["fd_include_mep_logit"] is True
        assert train["feature_node"] == "B"
        assert train["feature_weight"] == 200
        assert train["abort_on_nonfinite"] is True


def test_eps32_alpha8_pilot_grid() -> None:
    expected = {
        "pilot_mep_eps32_alpha8_logit10.yaml": ("mep_baseline", 0.0),
        "pilot_fd_eps32_alpha8_fw5.yaml": ("ours_fd", 5.0),
        "pilot_fd_eps32_alpha8_fw10.yaml": ("ours_fd", 10.0),
        "pilot_fd_eps32_alpha8_fw25.yaml": ("ours_fd", 25.0),
    }
    for name, (objective, feature_weight) in expected.items():
        config = load_config(ROOT / "configs" / "train" / name)
        train = config["train"]
        assert config["deterministic"] is True
        assert train["objective"] == objective
        assert train["backend"] == "mep"
        assert train["epochs"] == 40
        assert train["epsilon"] == 32
        assert train["alpha"] == 8
        assert train["lr"] == 0.1
        assert train["mep_logit_weight"] == 10
        assert train["feature_weight"] == feature_weight
        assert train["abort_on_nonfinite"] is True


def test_eps32_alpha8_high_fd_grid() -> None:
    for feature_weight in (50, 100, 200, 400):
        name = f"pilot_fd_eps32_alpha8_fw{feature_weight}.yaml"
        config = load_config(ROOT / "configs" / "train" / name)
        train = config["train"]
        assert config["deterministic"] is True
        assert train["objective"] == "ours_fd"
        assert train["backend"] == "mep"
        assert train["epochs"] == 40
        assert train["epsilon"] == 32
        assert train["alpha"] == 8
        assert train["lr"] == 0.1
        assert train["mep_logit_weight"] == 10
        assert train["feature_weight"] == feature_weight
        assert train["abort_on_nonfinite"] is True


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


def test_sweep_summary_accepts_epsilon_without_paper_target(tmp_path: Path) -> None:
    result = tmp_path / "evaluation.json"
    result.write_text(
        json.dumps(
            {
                "checkpoint_epoch": 42,
                "training_config": {
                    "data": {"dataset": "cifar10"},
                    "model": {"name": "resnet18"},
                    "seed": 0,
                    "train": {"objective": "ours_fd", "epsilon": 64},
                },
                "evaluation_config": {"epsilon": 64, "noise": {"enabled": False}},
                "metrics": {"clean": 0.75, "pgd50": 0.25},
            }
        ),
        encoding="utf-8",
    )
    csv_path, markdown_path = summarize_results([result], tmp_path / "sweep")
    assert "ours_fd,0,64.0,64.0,42,pgd50,25.0" in csv_path.read_text(encoding="utf-8")
    assert "ours_fd | 0 | 64/255 | 64/255 | 42 | 75.00 | 25.00" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_aaer_summary_requires_final_no_noise_matched_epsilon_and_aggregates(tmp_path: Path) -> None:
    results = []
    for seed, clean, pgd50 in ((0, 0.5, 0.2), (1, 0.6, 0.3), (2, 0.7, 0.4)):
        path = tmp_path / f"seed{seed}.json"
        path.write_text(
            json.dumps(
                {
                    "protocol": "aaer_official_pgd50_10",
                    "checkpoint_role": "final",
                    "training_config": {
                        "data": {"dataset": "cifar10"},
                        "model": {"name": "preactresnet18"},
                        "seed": seed,
                        "train": {"objective": "ours_fd", "epsilon": 8},
                    },
                    "evaluation_config": {
                        "epsilon": 8,
                        "step_size": 2,
                        "attacks": ["clean", "pgd50"],
                        "restarts": 10,
                        "freeze_misclassified": True,
                        "noise": {"enabled": False},
                    },
                    "metrics": {"clean": clean, "pgd50": pgd50},
                }
            ),
            encoding="utf-8",
        )
        results.append(path)
    csv_path, markdown_path = summarize_aaer_table2(results, tmp_path / "aaer")
    csv = csv_path.read_text(encoding="utf-8")
    assert "cifar10,preactresnet18,ours_fd,8,clean,3,\"0,1,2\",60.0,10.0,complete" in csv
    assert "60.00 ± 10.00" in markdown_path.read_text(encoding="utf-8")
