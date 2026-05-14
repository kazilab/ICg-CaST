"""I/O helpers for simulation outputs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from ._branding import COHORT_FILENAME
from .config import SimConfig


def ensure_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_simulation_metadata(cfg: SimConfig, outdir: str | Path, extra: dict[str, object] | None = None) -> Path:
    output_dir = ensure_dir(outdir)
    payload = asdict(cfg)
    payload["outdir"] = str(payload["outdir"])
    if extra:
        payload.update(extra)
    path = output_dir / "simulation_metadata.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def write_cohort(cohort: pd.DataFrame, outdir: str | Path, filename: str = COHORT_FILENAME) -> Path:
    output_dir = ensure_dir(outdir)
    path = output_dir / filename
    cohort.to_csv(path, index=False)
    return path
