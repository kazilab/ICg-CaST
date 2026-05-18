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
    table = calibration_metrics(np.asarray(y), np.asarray(proba), n_bins=n_bins)
    bins = table[table["bin"] != "summary"].sort_values("bin")
    mean_pred = bins["mean_predicted_risk"].to_numpy(dtype=float)
    obs_frac = bins["observed_event_rate"].to_numpy(dtype=float)
    counts = bins["n"].to_numpy(dtype=int)
    return mean_pred, obs_frac, counts
