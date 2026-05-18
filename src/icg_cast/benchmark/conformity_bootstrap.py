"""Bootstrap confidence intervals for intervention-conformity scores.

Operates on a fitted :class:`~icg_cast.bottleneck.MechanismBottleneckClassifier`
by resampling *test rows* with replacement and recomputing conformity fractions.
This quantifies uncertainty on the counterfactual audit rather than on
regression coefficients.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def responsive_conformity_from_table(
    mean_risk_change: np.ndarray,
    expected_direction_dgp: np.ndarray,
    *,
    threshold: float = 0.005,
) -> float:
    """Scalar responsive DGP conformity from aligned arrays (excluding exp_dir==0).

    The effective response threshold is ``max(threshold, 0.1 * sample_std)``
    over scored mean-risk changes. The 0.005 floor preserves the historical
    minimum practical effect size, while the dispersion term raises the bar
    when intervention effects vary widely across the audit panel.
    """
    return _directional_conformity_from_table(
        mean_risk_change,
        expected_direction_dgp,
        threshold=threshold,
    )


def _effective_response_threshold(deltas: np.ndarray, threshold: float) -> float:
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    if deltas.size <= 1:
        return float(threshold)
    sample_std = float(np.std(deltas, ddof=1))
    if not np.isfinite(sample_std):
        sample_std = 0.0
    return float(max(threshold, 0.1 * sample_std))


def _directional_conformity_from_table(
    mean_risk_change: np.ndarray,
    expected_direction: np.ndarray,
    *,
    threshold: float = 0.005,
) -> float:
    expected_direction = np.asarray(expected_direction, dtype=int)
    scored = expected_direction != 0
    if not scored.any():
        return float("nan")
    ok = responsive_passes_from_table(
        mean_risk_change,
        expected_direction,
        threshold=threshold,
    )[scored]
    return float(ok.mean())


def responsive_passes_from_table(
    mean_risk_change: np.ndarray,
    expected_direction: np.ndarray,
    *,
    threshold: float = 0.005,
) -> np.ndarray:
    """Per-intervention responsive pass flags using the adaptive threshold rule."""
    expected_direction = np.asarray(expected_direction, dtype=int)
    mean_risk_change = np.asarray(mean_risk_change, dtype=float)
    if mean_risk_change.shape != expected_direction.shape:
        raise ValueError("mean_risk_change and expected_direction must have matching shape")
    passes = np.zeros(mean_risk_change.shape, dtype=bool)
    scored = expected_direction != 0
    if not scored.any():
        return passes
    d = mean_risk_change[scored]
    e = expected_direction[scored]
    effective_threshold = _effective_response_threshold(d, threshold)
    passes[scored] = (np.sign(d) == np.sign(e)) & (np.abs(d) >= effective_threshold)
    return passes


def prior_conformity_from_table(
    mean_risk_change: np.ndarray,
    expected_direction: np.ndarray,
    *,
    threshold: float = 0.005,
) -> float:
    """Prior conformity with the same no-zero-pass response rule as DGP scoring."""
    return _directional_conformity_from_table(
        mean_risk_change,
        expected_direction,
        threshold=threshold,
    )


def dgp_conformity_from_table(
    mean_risk_change: np.ndarray,
    expected_direction_dgp: np.ndarray,
    *,
    threshold: float = 0.005,
) -> float:
    return prior_conformity_from_table(
        mean_risk_change, expected_direction_dgp, threshold=threshold,
    )


def _per_row_intervention_deltas(
    mb,
    X: pd.DataFrame,
    interventions: Mapping[str, Mapping[str, float | int]],
) -> dict[str, np.ndarray]:
    """Shape (n,): per-row mean risk change is not computed; we need per-row after-before.

    Returns mapping intervention_name -> delta vector of length n.
    """
    base = mb.predict_proba(X)[:, 1]
    out: dict[str, np.ndarray] = {}
    for name, spec in interventions.items():
        mb.clear_interventions()
        for unit, scale in spec.items():
            mb.intervene(unit=unit, scale=float(scale))
        after = mb.predict_proba(X)[:, 1]
        out[name] = after - base
    mb.clear_interventions()
    return out


def bootstrap_conformity(
    mb,
    X_test: pd.DataFrame,
    interventions: Mapping[str, Mapping[str, float | int]],
    expected_directions_prior: Mapping[str, int],
    expected_directions_dgp: Mapping[str, int],
    *,
    n_bootstrap: int = 300,
    random_state: int | None = 7,
    responsive_threshold: float = 0.005,
) -> dict[str, tuple[float, float, float]]:
    """Bootstrap (point, pct_low, pct_high) for prior, dgp, and responsive conformity.

    Percentiles are 2.5 and 97.5 (equal-tailed 95% interval). The *point* estimate
    uses the full test set (same as non-bootstrap run). Bootstrap draws resample
    rows of ``X_test`` and re-averages per-intervention Δ risk on the resample,
    then recomputes the three scalar scores.
    """
    order = list(interventions.keys())
    n = len(X_test)
    rng = np.random.default_rng(random_state)

    prior_dir = np.array([expected_directions_prior[k] for k in order], dtype=int)
    dgp_dir = np.array([expected_directions_dgp[k] for k in order], dtype=int)

    deltas_map = _per_row_intervention_deltas(mb, X_test, interventions)
    delta_mat = np.column_stack([deltas_map[k] for k in order])  # (n, n_int)

    # Point estimates: mean delta per intervention on full sample
    mean_full = delta_mat.mean(axis=0)
    pt_prior = prior_conformity_from_table(
        mean_full, prior_dir, threshold=responsive_threshold
    )
    pt_dgp = dgp_conformity_from_table(
        mean_full, dgp_dir, threshold=responsive_threshold
    )
    pt_resp = responsive_conformity_from_table(mean_full, dgp_dir, threshold=responsive_threshold)

    if n_bootstrap <= 0:
        return {
            "prior_conformity": (pt_prior, pt_prior, pt_prior),
            "dgp_conformity": (pt_dgp, pt_dgp, pt_dgp),
            "responsive_dgp_conformity": (pt_resp, pt_resp, pt_resp),
        }

    bp_prior: list[float] = []
    bp_dgp: list[float] = []
    bp_resp: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        means = delta_mat[idx].mean(axis=0)
        bp_prior.append(
            prior_conformity_from_table(means, prior_dir, threshold=responsive_threshold)
        )
        bp_dgp.append(
            dgp_conformity_from_table(means, dgp_dir, threshold=responsive_threshold)
        )
        bp_resp.append(
            responsive_conformity_from_table(means, dgp_dir, threshold=responsive_threshold)
        )

    def pct(xs: list[float], lo: float, hi: float) -> tuple[float, float]:
        arr = np.asarray(xs, dtype=float)
        return float(np.nanpercentile(arr, lo)), float(np.nanpercentile(arr, hi))

    pl, ph = pct(bp_prior, 2.5, 97.5)
    dl, dh = pct(bp_dgp, 2.5, 97.5)
    rl, rh = pct(bp_resp, 2.5, 97.5)

    return {
        "prior_conformity": (pt_prior, pl, ph),
        "dgp_conformity": (pt_dgp, dl, dh),
        "responsive_dgp_conformity": (pt_resp, rl, rh),
    }
