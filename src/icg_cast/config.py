"""Configuration and user-extensible archetype definitions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from .constants import KCC_NAMES


def _is_finite_number(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class ChemicalArchetype:
    """User-defined chemical mechanism vector.

    KCC values are synthetic mechanism priors in ``[0, 1]``. They are not
    measured carcinogenic potency estimates.
    """

    name: str
    kcc: Sequence[float]
    expected_signature: str = "mixed"
    dominant_aop: str = "user_defined"
    clip: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("archetype name must be non-empty")
        values = np.asarray(self.kcc, dtype=float)
        if values.size != len(KCC_NAMES):
            raise ValueError(f"kcc must contain exactly {len(KCC_NAMES)} values, got {values.size}")
        if not np.isfinite(values).all():
            raise ValueError("kcc values must be finite")
        if self.clip:
            values = np.clip(values, 0.0, 1.0)
        elif np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("kcc values must be in [0, 1]; pass clip=True to clip explicitly")
        object.__setattr__(self, "kcc", tuple(float(v) for v in values))


@dataclass(frozen=True)
class SimConfig:
    """Configuration for synthetic cohort simulation."""

    n: int = 1200
    months: int = 72
    seed: int = 7
    outdir: str | Path = "outputs"
    event_hazard_scale: float = 0.020
    archetype_prior: Mapping[str, float] | None = None
    dose_lognormal_mean: float = -0.10
    dose_lognormal_sigma: float = 0.75
    dose_min: float = 0.02
    dose_max: float = 6.0
    coefficient_mode: Literal["point", "prior_sample"] = "point"
    coefficient_seed: int | None = None
    simulator_backend: Literal["python", "vectorized"] = "python"
    retain_trajectories: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.n, int) or isinstance(self.n, bool) or self.n <= 0:
            raise ValueError("n must be a positive integer")
        if not isinstance(self.months, int) or isinstance(self.months, bool) or self.months <= 0:
            raise ValueError("months must be a positive integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        if self.coefficient_seed is not None and (
            not isinstance(self.coefficient_seed, int) or isinstance(self.coefficient_seed, bool)
        ):
            raise ValueError("coefficient_seed must be an integer or None")
        if not _is_finite_number(self.event_hazard_scale) or self.event_hazard_scale < 0:
            raise ValueError("event_hazard_scale must be non-negative")
        for name in ("dose_lognormal_mean", "dose_lognormal_sigma", "dose_min", "dose_max"):
            if not _is_finite_number(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.dose_lognormal_sigma < 0:
            raise ValueError("dose_lognormal_sigma must be non-negative")
        if self.dose_min <= 0 or self.dose_max <= self.dose_min:
            raise ValueError("dose_min and dose_max must be positive with dose_max > dose_min")
        if self.coefficient_mode not in ("point", "prior_sample"):
            raise ValueError("coefficient_mode must be 'point' or 'prior_sample'")
        if self.simulator_backend not in ("python", "vectorized"):
            raise ValueError("simulator_backend must be 'python' or 'vectorized'")
        if not isinstance(self.retain_trajectories, bool):
            raise ValueError("retain_trajectories must be a boolean")
        if self.archetype_prior is not None:
            if not isinstance(self.archetype_prior, Mapping):
                raise ValueError("archetype_prior must be a mapping of archetype name to weight")
            weights = np.asarray(list(self.archetype_prior.values()), dtype=float)
            if weights.size == 0:
                raise ValueError("archetype_prior must contain at least one weight")
            if not np.isfinite(weights).all():
                raise ValueError("archetype_prior weights must be finite")
            if np.any(weights < 0):
                raise ValueError("archetype_prior weights must be non-negative")
            if float(weights.sum()) <= 0:
                raise ValueError("archetype_prior must contain at least one positive weight")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")

    def resolved_coefficient_seed(self) -> int:
        """Return the seed used for coefficient-prior sampling."""
        return self.seed if self.coefficient_seed is None else self.coefficient_seed
