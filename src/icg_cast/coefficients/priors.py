"""Prior samplers for coefficient-card uncertainty.

The registry stores point values as prior centers. Evidence level supplies a
default spread: E1 is tightest, E5 widest. Card-level ``prior_params`` can
override the spread or bounds for a specific coefficient.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from scipy.special import expit, logit

from .registry import CoefficientCard, CoefficientRegistry, registry

EVIDENCE_LOG_SIGMA: dict[str, float] = {
    "E1": 0.05,
    "E2": 0.10,
    "E3": 0.18,
    "E4": 0.28,
    "E5": 0.45,
}

EVIDENCE_DIRICHLET_CONCENTRATION: dict[str, float] = {
    "E1": 2000.0,
    "E2": 1000.0,
    "E3": 300.0,
    "E4": 120.0,
    "E5": 40.0,
}


def prior_sigma_for_evidence(evidence_level: str) -> float:
    """Return the default log/normal-scale prior spread for an evidence level."""
    try:
        return EVIDENCE_LOG_SIGMA[evidence_level]
    except KeyError as exc:
        raise ValueError(f"unknown evidence level: {evidence_level!r}") from exc


def get_prior_params(evidence_level: str, median: float | int = 1.0) -> dict[str, float | str]:
    """Compatibility helper describing the default scalar prior for an evidence level."""
    sigma = prior_sigma_for_evidence(evidence_level)
    distribution = "lognormal" if float(median) > 0 else "signed_lognormal"
    if 0.0 < float(median) < 1.0:
        distribution = "logit_normal"
    return {"distribution": distribution, "sigma": sigma}


def _concentration_for_evidence(evidence_level: str) -> float:
    try:
        return EVIDENCE_DIRICHLET_CONCENTRATION[evidence_level]
    except KeyError as exc:
        raise ValueError(f"unknown evidence level: {evidence_level!r}") from exc


def _is_structural_fixed(card: CoefficientCard) -> bool:
    if card.is_string:
        return True
    name = card.name
    fixed_suffixes = (".seed", ".min", ".max", ".min_count")
    return name.endswith(fixed_suffixes)


def _params_float(params: dict[str, Any], key: str, default: float) -> float:
    value = params.get(key, default)
    return float(value)


def _normal_sample(
    values: np.ndarray,
    rng: np.random.Generator,
    *,
    sigma: float,
    params: dict[str, Any],
) -> np.ndarray:
    sd = params.get("sd")
    if sd is None:
        scale = _params_float(params, "scale", 1.0)
        sd_arr = np.maximum(np.abs(values) * sigma, sigma * scale)
    else:
        sd_arr = np.asarray(sd, dtype=float)
    sampled = rng.normal(values, sd_arr)
    low = params.get("low")
    high = params.get("high")
    if low is not None or high is not None:
        sampled = np.clip(
            sampled,
            -np.inf if low is None else float(low),
            np.inf if high is None else float(high),
        )
    return sampled


def _lognormal_sample(
    values: np.ndarray,
    rng: np.random.Generator,
    *,
    sigma: float,
    params: dict[str, Any],
) -> np.ndarray:
    sigma = _params_float(params, "sigma", sigma)
    sampled = values * np.exp(rng.normal(0.0, sigma, size=values.shape))
    low = params.get("low")
    high = params.get("high")
    if low is not None or high is not None:
        sampled = np.clip(
            sampled,
            -np.inf if low is None else float(low),
            np.inf if high is None else float(high),
        )
    return sampled


def _signed_lognormal_sample(
    values: np.ndarray,
    rng: np.random.Generator,
    *,
    sigma: float,
    params: dict[str, Any],
) -> np.ndarray:
    zero_mask = values == 0.0
    sampled = np.sign(values) * _lognormal_sample(
        np.abs(values), rng, sigma=sigma, params=params
    )
    if np.any(zero_mask):
        sampled[zero_mask] = _normal_sample(
            values[zero_mask], rng, sigma=sigma, params=params
        )
    return sampled


def _logit_normal_sample(
    values: np.ndarray,
    rng: np.random.Generator,
    *,
    sigma: float,
    params: dict[str, Any],
) -> np.ndarray:
    sigma = _params_float(params, "sigma", sigma)
    low = _params_float(params, "low", 0.0)
    high = _params_float(params, "high", 1.0)
    if high <= low:
        raise ValueError(f"logit_normal high must be greater than low; got {low}, {high}")
    eps = _params_float(params, "eps", 1e-6)
    scaled = np.clip((values - low) / (high - low), eps, 1.0 - eps)
    sampled = expit(logit(scaled) + rng.normal(0.0, sigma, size=values.shape))
    return low + sampled * (high - low)


def _dirichlet_sample(
    values: np.ndarray,
    rng: np.random.Generator,
    *,
    params: dict[str, Any],
    evidence_level: str,
) -> np.ndarray:
    if np.any(values < 0):
        raise ValueError("dirichlet prior requires non-negative values")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("dirichlet prior requires at least one positive value")
    probs = values / total
    concentration = _params_float(
        params,
        "concentration",
        _concentration_for_evidence(evidence_level),
    )
    alpha = np.maximum(probs * concentration, 1e-3)
    return rng.dirichlet(alpha)


def _auto_distribution(card: CoefficientCard, values: np.ndarray) -> str:
    if _is_structural_fixed(card):
        return "fixed"
    if values.ndim == 1 and values.size > 1:
        if np.all(values >= 0.0) and np.isclose(values.sum(), 1.0, rtol=1e-6, atol=1e-8):
            return "dirichlet"
        if np.all((0.0 <= values) & (values <= 1.0)):
            return "logit_normal"
        if np.all(values > 0.0):
            return "lognormal"
        if np.all(values < 0.0):
            return "signed_lognormal"
        return "normal"
    scalar = float(values.reshape(-1)[0])
    if 0.0 < scalar < 1.0:
        return "logit_normal"
    if scalar > 0.0:
        return "lognormal"
    if scalar < 0.0:
        return "signed_lognormal"
    return "normal"


def sample_card_prior(
    card: CoefficientCard,
    rng: np.random.Generator,
) -> float | int | str | tuple[float, ...]:
    """Draw one coefficient value from a card's prior distribution."""
    if _is_structural_fixed(card) or card.prior_distribution == "fixed":
        return card.default_value
    sigma = prior_sigma_for_evidence(card.evidence_level)
    params = dict(card.prior_params)
    value = card.default_value
    values = np.asarray(value if isinstance(value, tuple) else [value], dtype=float)
    distribution = (
        _auto_distribution(card, values)
        if card.prior_distribution == "auto"
        else card.prior_distribution
    )
    if distribution == "fixed":
        sampled = values
    elif distribution == "normal":
        sampled = _normal_sample(values, rng, sigma=sigma, params=params)
    elif distribution == "lognormal":
        if np.any(values < 0.0):
            raise ValueError(f"{card.name!r} has negative value but lognormal prior")
        sampled = _lognormal_sample(values, rng, sigma=sigma, params=params)
    elif distribution == "signed_lognormal":
        sampled = _signed_lognormal_sample(values, rng, sigma=sigma, params=params)
    elif distribution == "logit_normal":
        sampled = _logit_normal_sample(values, rng, sigma=sigma, params=params)
    elif distribution == "dirichlet":
        sampled = _dirichlet_sample(
            values, rng, params=params, evidence_level=card.evidence_level
        )
    else:  # registry validation should make this unreachable
        raise ValueError(f"unsupported prior distribution: {distribution!r}")

    if isinstance(value, tuple):
        return tuple(float(x) for x in sampled)
    return float(sampled.reshape(-1)[0])


def sample_coefficient(
    name: str,
    median: float | int,
    evidence_level: str = "E5",
    *,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> float:
    """Compatibility wrapper for sampling one scalar coefficient."""
    active_rng = np.random.default_rng(seed) if rng is None else rng
    card = CoefficientCard(
        name=name,
        default_value=float(median),
        evidence_level=evidence_level,
    )
    return float(sample_card_prior(card, active_rng))


def sampled_registry(
    base: CoefficientRegistry | None = None,
    *,
    seed: int | None = None,
) -> CoefficientRegistry:
    """Return a registry whose numeric point values are sampled from priors."""
    source = registry() if base is None else base
    rng = np.random.default_rng(seed)
    cards = [
        replace(card, default_value=sample_card_prior(card, rng))
        for card in source
    ]
    return CoefficientRegistry(cards, schema_version=source.schema_version)
