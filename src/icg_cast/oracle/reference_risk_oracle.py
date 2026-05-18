"""Frozen benchmark labelling oracle for ICg-CaST.

Version ``v1.0`` is the latent-risk structural equation.
It is used only as a benchmark oracle. Biological refinements belong in
``icg_cast.biology.biological_risk_equation``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

ORACLE_VERSION = "v1.0"

_ALIASES: dict[str, tuple[str, ...]] = {
    "dna_adducts": ("state_final_DNA_adducts", "DNA_adducts", "dna_adducts"),
    "ros": ("state_final_ROS", "ROS", "ros"),
    "inflammation": ("state_final_inflammation", "inflammation"),
    "epigenetic_age": ("state_final_epigenetic_age", "epigenetic_age"),
    "proliferation": ("state_final_proliferation", "proliferation"),
    "driver_count": (
        "state_final_driver_count_proxy",
        "driver_count_proxy",
        "driver_count",
    ),
    "clone_fraction": ("state_final_clone_fraction", "clone_fraction"),
    "immune_clearance": ("state_final_immune_clearance", "immune_clearance"),
}


def _column(
    data: pd.DataFrame | Mapping[str, Any] | None,
    kwargs: Mapping[str, Any],
    key: str,
    n: int | None,
    *,
    strict: bool = True,
) -> np.ndarray:
    aliases = _ALIASES[key]
    if isinstance(data, pd.DataFrame):
        for name in aliases:
            if name in data.columns:
                return np.clip(data[name].to_numpy(dtype=float), 0.0, None)
        if strict:
            raise KeyError(
                f"missing state column for {key!r}; expected one of {aliases!r}"
            )
        return np.zeros(len(data), dtype=float)

    source: Mapping[str, Any] = kwargs if data is None else {**dict(data), **dict(kwargs)}
    for name in aliases:
        if name in source:
            arr = np.asarray(source[name], dtype=float)
            if arr.ndim == 0:
                arr = np.repeat(float(arr), 1 if n is None else n)
            return np.clip(arr, 0.0, None)
    if strict:
        raise KeyError(f"missing state value for {key!r}; expected one of {aliases!r}")
    return np.zeros(1 if n is None else n, dtype=float)


def _infer_n(data: pd.DataFrame | Mapping[str, Any] | None, kwargs: Mapping[str, Any]) -> int | None:
    if isinstance(data, pd.DataFrame):
        return len(data)
    source: Mapping[str, Any] = kwargs if data is None else {**dict(data), **dict(kwargs)}
    for value in source.values():
        arr = np.asarray(value)
        if arr.ndim > 0:
            return int(arr.size)
    return None


def reference_risk_oracle(
    states: pd.DataFrame | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> np.ndarray | float:
    """Return frozen v1.0 oracle risk for final qAOP state values.

    ``states`` may be a DataFrame with ``state_final_*`` columns, a mapping of
    state names to values, or omitted in favour of keyword arguments.

    Bounded state variables, including ``clone_fraction``, enter linearly by
    design. Burden/count variables use ``log1p`` to compress unbounded scales.
    """
    n = _infer_n(states, kwargs)
    dna = _column(states, kwargs, "dna_adducts", n)
    ros = _column(states, kwargs, "ros", n)
    inflammation = _column(states, kwargs, "inflammation", n)
    epigenetic_age = _column(states, kwargs, "epigenetic_age", n)
    proliferation = _column(states, kwargs, "proliferation", n)
    drivers = _column(states, kwargs, "driver_count", n)
    clone = _column(states, kwargs, "clone_fraction", n)
    immune = _column(states, kwargs, "immune_clearance", n)

    risk = expit(
        -7.5
        + 1.20 * np.log1p(dna)
        + 0.70 * np.log1p(ros)
        + 0.95 * np.log1p(inflammation)
        + 1.50 * proliferation
        + 0.85 * np.log1p(epigenetic_age)
        + 1.80 * np.log1p(drivers)
        + 4.50 * clone
        - 0.90 * immune
    )
    if risk.size == 1 and not isinstance(states, pd.DataFrame):
        return float(risk[0])
    return risk


def get_oracle_version() -> str:
    """Return the frozen oracle version."""
    return ORACLE_VERSION
