"""Bootstrap confidence intervals for intervention-conformity scores.

Operates on a fitted :class:`~icg_cast.bottleneck.MechanismBottleneckClassifier`
by resampling *test rows* with replacement and recomputing conformity fractions.
This matches the Milestone 5.5 requirement for uncertainty quantification on the
counterfactual audit rather than on regression coefficients.
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
    """Scalar responsive DGP conformity from aligned arrays (excluding exp_dir==0)."""
    expected_direction_dgp = np.asarray(expected_direction_dgp, dtype=int)
    mean_risk_change = np.asarray(mean_risk_change, dtype=float)
    scored = expected_direction_dgp != 0
    if not scored.any():
        return float("nan")
    d = mean_risk_change[scored]
    e = expected_direction_dgp[scored]
    ok = (np.sign(d) == np.sign(e)) & (np.abs(d) >= threshold)
    return float(ok.mean())


def prior_conformity_from_table(
    mean_risk_change: np.ndarray,
    expected_direction: np.ndarray,
    *,
    atol: float = 1e-3,
) -> float:
    expected_direction = np.asarray(expected_direction, dtype=int)
    mean_risk_change = np.asarray(mean_risk_change, dtype=float)
    scored = expected_direction != 0
    if not scored.any():
        return float("nan")
    d = mean_risk_change[scored]
    e = expected_direction[scored]
    ok = (np.sign(d) == np.sign(e)) | np.isclose(d, 0.0, atol=atol)
    return float(ok.mean())


def dgp_conformity_from_table(
    mean_risk_change: np.ndarray,
    expected_direction_dgp: np.ndarray,
    *,
    atol: float = 1e-3,
) -> float:
    return prior_conformity_from_table(
        mean_risk_change, expected_direction_dgp, atol=atol,
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
    pt_prior = prior_conformity_from_table(mean_full, prior_dir)
    pt_dgp = dgp_conformity_from_table(mean_full, dgp_dir)
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
        bp_prior.append(prior_conformity_from_table(means, prior_dir))
        bp_dgp.append(dgp_conformity_from_table(means, dgp_dir))
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
