"""I/O helpers for simulation outputs."""

from __future__ import annotations

import json
import tempfile
import warnings
from dataclasses import asdict
from os import PathLike
from pathlib import Path

import pandas as pd

from ._branding import COHORT_FILENAME
from .config import SimConfig


def ensure_dir(
    path: str | PathLike[str],
    *,
    fallback_prefix: str | None = None,
) -> Path:
    """Create a writable output directory, optionally falling back to tempdir.

    A directory that exists but cannot be written to is treated the same as a
    creation failure. When ``fallback_prefix`` is supplied, a persistent
    temporary directory is created and returned instead of raising.
    """
    output_dir = Path(path)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        probe = output_dir / ".icg_cast_write_test"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        return output_dir
    except OSError as exc:
        if fallback_prefix is not None:
            fallback = Path(tempfile.mkdtemp(prefix=fallback_prefix))
            warnings.warn(
                f"could not create or write output directory {output_dir!s}: {exc}; "
                f"using temporary directory {fallback!s}",
                RuntimeWarning,
                stacklevel=2,
            )
            return fallback
        raise OSError(
            f"could not create or write output directory {output_dir!s}: {exc}"
        ) from exc


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
