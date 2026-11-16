"""Shared utilities: seeding, device selection, logging, timing, config, run records."""

from __future__ import annotations

import json
import logging
import platform
import random
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- seeds / device
def set_seed(seed: int) -> None:
    """Seed ``random``, ``numpy`` and ``torch`` (CPU, CUDA and MPS) for reproducibility."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: str | None = None) -> str:
    """Return ``"cuda"``, ``"mps"`` or ``"cpu"`` (in that order) unless ``prefer`` is given."""
    import torch

    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------- logging
def get_logger(name: str = "piiguard", level: int = logging.INFO) -> logging.Logger:
    """Return a console logger with a compact format (idempotent)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# --------------------------------------------------------------------------- timing
@dataclass
class Timer:
    """Wall-clock timer; use as a context manager and read ``elapsed`` (seconds)."""

    elapsed: float = 0.0
    _start: float = field(default=0.0, repr=False)

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self._start


@contextmanager
def timed() -> Iterator[Timer]:
    """Context manager yielding a :class:`Timer`."""
    with Timer() as t:
        yield t


# --------------------------------------------------------------------------- config
def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config; relative ``paths`` entries resolve against the project root."""
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = PROJECT_ROOT / p
    with open(p) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(p)
    for k, v in cfg.get("paths", {}).items():
        vp = Path(v)
        cfg["paths"][k] = str(vp if vp.is_absolute() else PROJECT_ROOT / vp)
    return cfg


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- run records
def git_commit() -> str:
    """Return the current git commit hash (``+dirty`` if uncommitted changes), or ``"unknown"``."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
        return sha + ("+dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_run_record(
    results_dir: str | Path,
    name: str,
    *,
    config: dict[str, Any],
    seed: int,
    metrics: dict[str, Any],
    timings: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a JSON run record (config, seed, git commit, device, timings, metrics).

    Only values passed in ``metrics`` are recorded; nothing is invented here.
    """
    out_dir = ensure_dir(results_dir)
    record = {
        "name": name,
        "timestamp": datetime.now(UTC).isoformat(),
        "seed": seed,
        "git_commit": git_commit(),
        "device": get_device(),
        "platform": platform.platform(),
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
        "timings": timings or {},
        "metrics": metrics,
        **(extra or {}),
    }
    path = out_dir / f"{name}_seed{seed}.json"
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=_json_default)
    return path


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")
