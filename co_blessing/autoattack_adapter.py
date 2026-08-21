from __future__ import annotations

import importlib
import inspect
import sys
import textwrap
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from .utils import REPO_ROOT, sha256_tree


REFERENCE_ROOT = REPO_ROOT / "related_code" / "ConvergeSmooth"
AUTOATTACK_ROOT = REFERENCE_ROOT / "autoattack"
EXPECTED_SHA256 = "aeb3b5167a3e4971af0fb0192733cff9b8e5bba79ef5722dd1a1fe576db1afc0"


def _zero_gradients_compat(inputs: Any) -> None:
    """PyTorch's removed ``torch.autograd.gradcheck.zero_gradients`` helper."""

    if isinstance(inputs, torch.Tensor):
        if inputs.grad is not None:
            inputs.grad.detach_()
            inputs.grad.zero_()
    elif isinstance(inputs, Iterable):
        for item in inputs:
            _zero_gradients_compat(item)


def _install_torch_compatibility_shims() -> None:
    # The pinned 2019 AutoAttack FAB implementation imports this private
    # helper, which was removed from recent PyTorch releases. Injecting its
    # original behavior keeps the reference source tree immutable and pinned.
    gradcheck = importlib.import_module("torch.autograd.gradcheck")
    if not hasattr(gradcheck, "zero_gradients"):
        gradcheck.zero_gradients = _zero_gradients_compat


def _patch_fab_device_compatibility() -> None:
    """Patch only device placement in the hash-pinned FAB implementation."""

    fab_module = importlib.import_module("autoattack.fab_pt")
    original = fab_module.FABAttack.projection_linf
    if getattr(original, "_co_blessing_device_compat", False):
        return
    source = textwrap.dedent(inspect.getsource(original))
    replacements = {
        "u = torch.arange(0, w.shape[0])":
            "u = torch.arange(0, w.shape[0], device=w.device)",
        "lb = torch.zeros(c2.shape[0])":
            "lb = torch.zeros(c2.shape[0], device=w.device)",
        "ub = torch.ones(c2.shape[0]) * (w.shape[1] - 1)":
            "ub = torch.ones(c2.shape[0], device=w.device) * (w.shape[1] - 1)",
        "counter2 = torch.zeros(lb.shape).long()":
            "counter2 = torch.zeros(lb.shape, device=w.device).long()",
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError(f"Unexpected pinned FAB source; missing line: {old}")
        source = source.replace(old, new)
    namespace = dict(original.__globals__)
    exec(compile(source, str(AUTOATTACK_ROOT / "fab_pt.py"), "exec"), namespace)
    patched = namespace["projection_linf"]
    patched._co_blessing_device_compat = True
    fab_module.FABAttack.projection_linf = patched


def source_metadata() -> dict[str, Any]:
    if not AUTOATTACK_ROOT.is_dir():
        return {"source": str(AUTOATTACK_ROOT), "available": False, "sha256": None}
    actual = sha256_tree(AUTOATTACK_ROOT)
    return {
        "source": str(AUTOATTACK_ROOT),
        "available": True,
        "sha256": actual,
        "expected_sha256": EXPECTED_SHA256,
        "matches_expected": actual == EXPECTED_SHA256,
    }


def _autoattack_class() -> type:
    if not AUTOATTACK_ROOT.is_dir():
        raise FileNotFoundError(
            "Reference AutoAttack snapshot is missing. Expected: " f"{AUTOATTACK_ROOT}"
        )
    metadata = source_metadata()
    if not metadata["matches_expected"]:
        raise RuntimeError(
            "Reference AutoAttack snapshot differs from the version used by the reproduction: "
            f"expected {EXPECTED_SHA256}, got {metadata['sha256']}"
        )
    _install_torch_compatibility_shims()
    reference = str(REFERENCE_ROOT)
    if reference not in sys.path:
        sys.path.insert(0, reference)
    module = importlib.import_module("autoattack")
    _patch_fab_device_compatibility()
    module_path = Path(module.__file__).resolve()
    if AUTOATTACK_ROOT.resolve() not in module_path.parents:
        raise RuntimeError(
            f"Imported AutoAttack from {module_path}, not the reference snapshot {AUTOATTACK_ROOT}"
        )
    return module.AutoAttack


def generate_autoattack(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    epsilon: float,
    batch_size: int,
    seed: int,
    attack: str,
    device: torch.device,
    log_path: Path,
) -> torch.Tensor:
    AutoAttack = _autoattack_class()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if attack == "aa":
        adversary = AutoAttack(
            model,
            norm="Linf",
            eps=epsilon,
            seed=seed,
            verbose=True,
            version="standard",
            device=str(device),
            log_path=str(log_path),
        )
    elif attack == "apgd-t":
        adversary = AutoAttack(
            model,
            norm="Linf",
            eps=epsilon,
            seed=seed,
            verbose=True,
            attacks_to_run=["apgd-t"],
            version="custom",
            device=str(device),
            log_path=str(log_path),
        )
    else:
        raise ValueError(f"Unsupported AutoAttack mode: {attack}")
    # Keeping x_orig on CPU matches the bundled reference evaluator and avoids
    # retaining the full CIFAR-10 set on GPU between attacks.
    return adversary.run_standard_evaluation(inputs.cpu(), targets.cpu(), bs=batch_size)
