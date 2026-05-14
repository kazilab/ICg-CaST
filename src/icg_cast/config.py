"""Configuration and user-extensible archetype definitions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from .constants import KCC_NAMES


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
        values = np.asarray(self.kcc, dtype=float)
        if values.size != len(KCC_NAMES):
            raise ValueError(f"kcc must contain exactly {len(KCC_NAMES)} values, got {values.size}")
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
    metadata: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if self.n <= 0:
            raise ValueError("n must be positive")
        if self.months <= 0:
            raise ValueError("months must be positive")
        if self.event_hazard_scale < 0:
            raise ValueError("event_hazard_scale must be non-negative")
        if self.dose_min <= 0 or self.dose_max <= self.dose_min:
            raise ValueError("dose_min and dose_max must be positive with dose_max > dose_min")
        if self.coefficient_mode not in ("point", "prior_sample"):
            raise ValueError("coefficient_mode must be 'point' or 'prior_sample'")

    def resolved_coefficient_seed(self) -> int:
        """Return the seed used for coefficient-prior sampling."""
        return self.seed if self.coefficient_seed is None else self.coefficient_seed
