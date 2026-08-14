from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed every RNG used by the reproduction code.

    The reference repositories seed NumPy and Torch but omit Python's RNG.  We
    seed it as well because torchvision augmentation and our orchestration code
    must be repeatable.  Deterministic algorithms remain opt-in so the default
    stays close to the original PyTorch behavior.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested ({value}) but CUDA is unavailable")
    return device


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def append_csv(path: Path, row: dict[str, Any], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def sha256_tree(root: Path, pattern: str = "*.py") -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob(pattern)):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_revision() -> str | None:
    return git_revision_at(REPO_ROOT)


def git_revision_at(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_metadata() -> dict[str, Any]:
    cuda_device = None
    if torch.cuda.is_available():
        cuda_device = torch.cuda.get_device_name(torch.cuda.current_device())
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": _package_version("torchvision"),
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_device": cuda_device,
        "git_revision": git_revision(),
        "reference_revisions": {
            "FGSM-PGI": git_revision_at(REPO_ROOT / "related_code" / "FGSM-PGI"),
            "ConvergeSmooth": git_revision_at(REPO_ROOT / "related_code" / "ConvergeSmooth"),
        },
    }


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def load_model_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load checkpoints saved with or without DataParallel's ``module.`` prefix."""

    try:
        model.load_state_dict(state)
        return
    except RuntimeError:
        pass
    cleaned = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(cleaned)
