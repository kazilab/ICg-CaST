"""Registry-backed biological risk equation.

This function intentionally shares the oracle's functional form, but its
coefficients are loaded from the active coefficient registry and can therefore
change under prior sampling or calibration.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

from icg_cast.coefficients import registry as _registry
from icg_cast.coefficients import sampled_registry, use_registry
from icg_cast.oracle.reference_risk_oracle import _column, _infer_n

BIOLOGY_VERSION = "v0.2-registry-backed"


def _coefficients() -> dict[str, float]:
    r = _registry()
    return {
        "intercept": r.get("dynamics.latent_risk.intercept"),
        "dna": r.get("dynamics.latent_risk.dna_coupling"),
        "ros": r.get("dynamics.latent_risk.ros_coupling"),
        "inflammation": r.get("dynamics.latent_risk.inflammation_coupling"),
        "proliferation": r.get("dynamics.latent_risk.proliferation_coupling"),
        "epigenetic_age": r.get("dynamics.latent_risk.epigenetic_coupling"),
        "driver": r.get("dynamics.latent_risk.driver_coupling"),
        "clone": r.get("dynamics.latent_risk.clone_coupling"),
        "immune": r.get("dynamics.latent_risk.immune_coupling"),
    }


def biological_risk_equation(
    states: pd.DataFrame | Mapping[str, Any] | None = None,
    *,
    use_priors: bool = False,
    seed: int | None = None,
    **kwargs: Any,
) -> np.ndarray | float:
    """Return biological risk for final qAOP state values.

    Args:
        states: DataFrame with ``state_final_*`` columns or a mapping of state
            names to values. Omit it and pass state values as keyword args for
            scalar calls.
        use_priors: Draw one seedable coefficient-prior realization before
            evaluating the equation.
        seed: Seed used when ``use_priors=True``.

    Omitted state values are treated as zero at this biological boundary. The
    ``clone_fraction`` term follows the frozen functional form and remains
    linear because clone fraction is defined on ``[0, 1]``.
    """
    if use_priors:
        with use_registry(sampled_registry(seed=seed)):
            return biological_risk_equation(states, use_priors=False, **kwargs)

    n = _infer_n(states, kwargs)
    c = _coefficients()
    dna = _column(states, kwargs, "dna_adducts", n, strict=False)
    ros = _column(states, kwargs, "ros", n, strict=False)
    inflammation = _column(states, kwargs, "inflammation", n, strict=False)
    epigenetic_age = _column(states, kwargs, "epigenetic_age", n, strict=False)
    proliferation = _column(states, kwargs, "proliferation", n, strict=False)
    drivers = _column(states, kwargs, "driver_count", n, strict=False)
    clone = _column(states, kwargs, "clone_fraction", n, strict=False)
    immune = _column(states, kwargs, "immune_clearance", n, strict=False)

    risk = expit(
        c["intercept"]
        + c["dna"] * np.log1p(dna)
        + c["ros"] * np.log1p(ros)
        + c["inflammation"] * np.log1p(inflammation)
        + c["proliferation"] * proliferation
        + c["epigenetic_age"] * np.log1p(epigenetic_age)
        + c["driver"] * np.log1p(drivers)
        + c["clone"] * clone
        + c["immune"] * immune
    )
    if risk.size == 1 and not isinstance(states, pd.DataFrame):
        return float(risk[0])
    return risk


def get_biology_version() -> str:
    """Return the biological equation version."""
    return BIOLOGY_VERSION
