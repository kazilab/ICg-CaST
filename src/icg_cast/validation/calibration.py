"""Predictive calibration helpers.

The legacy entry point :func:`icg_cast.models.calibration_metrics` returns a
long-form bin table plus a summary row. This module re-exports that helper
and adds two leaner helpers:

- :func:`expected_calibration_error` returns the scalar ECE only.
- :func:`calibration_curve` returns ``(mean_predicted, observed_fraction)``
  arrays suitable for plotting reliability diagrams.
"""

from __future__ import annotations

import numpy as np

from ..models import calibration_metrics

__all__ = [
    "calibration_curve",
    "calibration_metrics",
    "expected_calibration_error",
]


def expected_calibration_error(
    y: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Return the ECE scalar from :func:`icg_cast.models.calibration_metrics`."""
    table = calibration_metrics(np.asarray(y), np.asarray(proba), n_bins=n_bins)
    summary = table[table["bin"] == "summary"]
    if summary.empty:
        return float("nan")
    return float(summary["expected_calibration_error"].iloc[0])


def calibration_curve(
    y: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-bin (mean_predicted, observed_fraction, n) for a reliability plot."""
    y = np.asarray(y, dtype=int)
    proba = np.asarray(proba, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_id = np.clip(np.digitize(proba, bins, right=True) - 1, 0, n_bins - 1)
    mean_pred = np.full(n_bins, np.nan)
    obs_frac = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        mask = bin_id == i
        n = int(mask.sum())
        counts[i] = n
        if n:
            mean_pred[i] = float(proba[mask].mean())
            obs_frac[i] = float(y[mask].mean())
    return mean_pred, obs_frac, counts
