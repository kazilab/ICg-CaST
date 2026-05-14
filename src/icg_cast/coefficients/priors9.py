"""
Milestone 9: Coefficient Priors and Uncertainty
Seedable samplers for coefficient distributions based on evidence level.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def get_prior_params(evidence_level: str, median: float) -> dict[str, Any]:
    """
    Return prior distribution and parameters based on evidence level.
    
    E1: Very tight (published quantitative) → narrow normal
    E2: Tight (published qualitative)
    E3: Moderate (AOP weight-of-evidence)
    E4: Wide (expert estimate)
    E5: Very wide (hand-tuned baseline)
    """
    if evidence_level == "E1":
        return {"distribution": "normal", "loc": median, "scale": abs(median) * 0.05}
    elif evidence_level == "E2":
        return {"distribution": "normal", "loc": median, "scale": abs(median) * 0.12}
    elif evidence_level == "E3":
        return {"distribution": "normal", "loc": median, "scale": abs(median) * 0.25}
    elif evidence_level == "E4":
        return {"distribution": "lognormal", "median": median, "sigma": 0.35}
    else:  # E5 - widest
        return {"distribution": "lognormal", "median": median, "sigma": 0.55}


def sample_coefficient(
    name: str,
    median: float,
    evidence_level: str = "E5",
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> float:
    """
    Sample a coefficient from its prior distribution.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    
    params = get_prior_params(evidence_level, median)
    
    if params["distribution"] == "normal":
        value = rng.normal(params["loc"], params["scale"])
        # Ensure positivity for rate-like parameters
        if median > 0:
            value = max(0.01 * median, value)
        return float(value)
    
    elif params["distribution"] == "lognormal":
        # lognormal parameterized by median and sigma
        mu = np.log(params["median"])
        sigma = params["sigma"]
        value = rng.lognormal(mu, sigma)
        return float(value)
    
    else:
        raise ValueError(f"Unknown distribution: {params['distribution']}")


def sample_coefficient_vector(
    names: list[str],
    medians: list[float],
    evidence_levels: list[str],
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Sample multiple coefficients at once."""
    if rng is None:
        rng = np.random.default_rng(seed)
    
    values = []
    for name, median, ev in zip(names, medians, evidence_levels, strict=False):
        val = sample_coefficient(name, median, ev, rng=rng)
        values.append(val)
    return np.array(values)


# === Milestone 9 Extension: Correlated Priors ===

def sample_correlated_coefficients(
    names: list[str],
    medians: list[float],
    evidence_levels: list[str],
    correlation: float = 0.6,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """
    Sample correlated coefficients (e.g., DNA adduct decay and mutation rate).
    
    Uses a simple Gaussian copula approach for correlation.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    
    n = len(names)
    if n < 2:
        return sample_coefficient_vector(names, medians, evidence_levels, rng=rng)
    
    # Generate correlated standard normals
    cov = np.full((n, n), correlation)
    np.fill_diagonal(cov, 1.0)
    z = rng.multivariate_normal(np.zeros(n), cov)
    
    values = []
    for i, (_name, median, ev) in enumerate(zip(names, medians, evidence_levels, strict=False)):
        params = get_prior_params(ev, median)
        
        if params["distribution"] == "lognormal":
            # Transform correlated normal to lognormal
            sigma = params["sigma"]
            mu = np.log(params["median"])
            val = np.exp(mu + sigma * z[i])
        else:
            # Normal case
            val = params["loc"] + params["scale"] * z[i]
            if median > 0:
                val = max(0.01 * median, val)
        
        values.append(float(val))
    
    return np.array(values)
