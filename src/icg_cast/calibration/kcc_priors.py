"""ToxCast-driven KCC prior calibration."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import KCC_NAMES
from ..data_sources.common import DataSourceBundle, read_local_table

_KCC_ID_PATTERN = re.compile(r"KCC\s*(\d+)", re.IGNORECASE)


def _resolve_kcc_index(value: object) -> int | None:
    """Resolve a user-provided KCC identifier to a 0-based KCC_NAMES index."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = _KCC_ID_PATTERN.fullmatch(text) or _KCC_ID_PATTERN.search(text)
    if match:
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(KCC_NAMES):
            return idx
        return None
    if text in KCC_NAMES:
        return KCC_NAMES.index(text)
    return None


def calibrate_kcc_priors_from_toxcast(
    toxcast_bundle: DataSourceBundle,
    mapping: pd.DataFrame | str | Path,
    chemical_column: str = "chemical_id",
    assay_column: str = "assay",
    hitcall_column: str = "hit_call",
    mapping_assay_column: str = "assay",
    mapping_kcc_column: str = "kcc_id",
) -> dict[str, tuple[float, ...]]:
    """Build per-chemical KCC priors from ToxCast hit calls and an assay→KCC mapping.

    For each chemical, the KCCi prior is the fraction of mapped assays
    (assay→KCCi) that registered a hit. KCCs with no mapped assays default to
    0.0. The result is a dict suitable for use as the ``archetype_kcc`` field
    of a :class:`CalibrationBundle`, with chemical identifiers acting as
    archetype names.
    """
    if toxcast_bundle.metadata.get("adapter") != "toxcast":
        raise ValueError("calibrate_kcc_priors_from_toxcast expects a ToxCast adapter bundle")
    data = toxcast_bundle.data.copy()
    missing = {chemical_column, assay_column, hitcall_column} - set(data.columns)
    if missing:
        raise KeyError(f"ToxCast bundle missing required columns: {sorted(missing)}")

    if isinstance(mapping, pd.DataFrame):
        mmap = mapping.copy()
    else:
        mmap = read_local_table(mapping)
    map_missing = {mapping_assay_column, mapping_kcc_column} - set(mmap.columns)
    if map_missing:
        raise KeyError(f"mapping missing required columns: {sorted(map_missing)}")

    mmap = mmap[[mapping_assay_column, mapping_kcc_column]].rename(
        columns={mapping_assay_column: "_assay", mapping_kcc_column: "_kcc"}
    )
    mmap["_kcc_index"] = mmap["_kcc"].map(_resolve_kcc_index)
    mmap = mmap.dropna(subset=["_kcc_index"]).copy()
    mmap["_kcc_index"] = mmap["_kcc_index"].astype(int)
    if mmap.empty:
        raise ValueError("no rows in mapping resolve to a recognised KCC index")

    data = data.rename(
        columns={chemical_column: "_chem", assay_column: "_assay", hitcall_column: "_hit"}
    )
    data["_hit"] = pd.to_numeric(data["_hit"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    merged = data.merge(mmap[["_assay", "_kcc_index"]], on="_assay", how="inner")
    if merged.empty:
        raise ValueError("no assays shared between ToxCast bundle and mapping")

    priors: dict[str, tuple[float, ...]] = {}
    for chem, sub in merged.groupby("_chem"):
        vec = np.zeros(len(KCC_NAMES), dtype=float)
        for idx, sub2 in sub.groupby("_kcc_index"):
            vec[int(idx)] = float(sub2["_hit"].mean())
        priors[str(chem)] = tuple(float(np.clip(v, 0.0, 1.0)) for v in vec)
    return priors
