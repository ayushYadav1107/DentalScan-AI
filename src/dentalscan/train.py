"""Reproducible training runs with recorded provenance.

Every run writes a ``run_meta.json`` next to its weights containing the seed,
the full resolved hyperparameter set, library versions and the git commit. A
result you cannot attribute to an exact configuration is not a result, and the
single hardest thing to reconstruct three months later is which of nine similar
runs produced the number in the README.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

from .constants import ROOT, RUNS_DIR


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for module in ("torch", "ultralytics", "numpy"):
        try:
            out[module] = __import__(module).__version__
        except Exception:
            out[module] = "unavailable"
    try:
        import torch
        out["cuda"] = torch.version.cuda or "cpu"
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        out["cuda"] = out["gpu"] = "unavailable"
    return out


# Baseline recipe: the configuration the original project trained with, kept
# verbatim so every ablation is measured against a real, previously-reported run.
BASELINE_HYPERPARAMS: dict[str, Any] = {
    "epochs": 250,
    "imgsz": 640,
    "batch": 8,
    "patience": 100,
    "optimizer": "auto",
    "lr0": 0.01,
    "lrf": 0.01,
    "cos_lr": False,
    "close_mosaic": 10,
    # Geometric augmentation
    "degrees": 10.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 5.0,
    "perspective": 0.0005,
    "flipud": 0.2,
    "fliplr": 0.5,
    # Photometric augmentation
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    # Composition augmentation
    "mosaic": 1.0,
    "mixup": 0.2,
    "copy_paste": 0.1,
}

# Named groups referenced by the ablation configs. Setting a group "off" means
# assigning each of its parameters the value that disables it.
AUGMENTATION_GROUPS: dict[str, dict[str, float]] = {
    "geometric": {"degrees": 0.0, "translate": 0.0, "scale": 0.0,
                  "shear": 0.0, "perspective": 0.0},
    "flips": {"flipud": 0.0, "fliplr": 0.0},
    "photometric": {"hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0},
    "mosaic": {"mosaic": 0.0},
    "mixup": {"mixup": 0.0},
    "copy_paste": {"copy_paste": 0.0},
}


@dataclass
class RunConfig:
    """One training run, fully specified."""

    name: str
    model: str = "yolov10s.pt"
    data: str = "configs/data.yaml"
    seed: int = 0
    device: str = "0"
    project: str = str(RUNS_DIR / "experiments")
    hyperparams: dict[str, Any] = field(default_factory=lambda: dict(BASELINE_HYPERPARAMS))
    notes: str = ""

    def resolved(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "data": self.data,
            "seed": self.seed,
            "device": self.device,
            **self.hyperparams,
        }


def apply_ablation(
    base: dict[str, Any],
    disable_groups: list[str] | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive a hyperparameter set from the baseline by turning things off.

    Ablations are expressed as *deltas from the baseline*, never as fresh
    configs, so a difference in results can only come from the delta.
    """
    params = dict(base)
    for group in disable_groups or []:
        if group not in AUGMENTATION_GROUPS:
            raise KeyError(
                f"Unknown augmentation group '{group}'. "
                f"Known: {sorted(AUGMENTATION_GROUPS)}"
            )
        params.update(AUGMENTATION_GROUPS[group])
    params.update(overrides or {})
    return params


def train_once(config: RunConfig, deterministic: bool = True) -> Path:
    """Train one model and return the directory holding its weights and metadata."""
    from ultralytics import YOLO

    os.environ.setdefault("PYTHONHASHSEED", str(config.seed))
    if deterministic:
        # cuBLAS needs this set before the first CUDA context to make matmuls
        # reproducible; without it two runs with the same seed can still differ.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    started = time.time()
    model = YOLO(config.model)
    model.train(
        data=config.data,
        name=config.name,
        project=config.project,
        seed=config.seed,
        deterministic=deterministic,
        device=config.device,
        exist_ok=True,
        plots=True,
        val=True,
        **config.hyperparams,
    )
    duration = time.time() - started

    run_dir = Path(config.project) / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_meta.json").write_text(json.dumps({
        "config": asdict(config),
        "resolved_hyperparams": config.resolved(),
        "git_commit": _git_commit(),
        "versions": _versions(),
        "wall_clock_seconds": round(duration, 1),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2))
    return run_dir


def weights_path(run_dir: Path, which: str = "best") -> Path:
    path = Path(run_dir) / "weights" / f"{which}.pt"
    if not path.exists():
        raise FileNotFoundError(f"No weights at {path}")
    return path
