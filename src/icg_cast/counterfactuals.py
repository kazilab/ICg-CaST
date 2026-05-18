"""Feature-space counterfactual stress tests for baseline models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .interventions import COUNTERFACTUAL_FEATURE_INTERVENTIONS


def counterfactual_tests(model, test_df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Apply mechanism-specific feature perturbations and summarize risk changes."""
    base_x = test_df[cols].copy()
    base_pred = model.predict_proba(base_x)[:, 1]
    tests = []

    for name, spec in COUNTERFACTUAL_FEATURE_INTERVENTIONS.items():
        x_cf = base_x.copy()
        for col, factor in spec.column_scales.items():
            if col in x_cf.columns:
                x_cf[col] = x_cf[col] * float(factor)
        cf_pred = model.predict_proba(x_cf)[:, 1]
        delta = cf_pred - base_pred
        expected = int(spec.expected_direction)
        observed = int(np.sign(float(np.mean(delta))))
        tests.append(
            {
                "intervention": name,
                "mean_predicted_risk_before": float(np.mean(base_pred)),
                "mean_predicted_risk_after": float(np.mean(cf_pred)),
                "mean_absolute_risk_change": float(np.mean(delta)),
                "median_absolute_risk_change": float(np.median(delta)),
                "intervention_severity_weight": float(spec.resolved_severity_weight()),
                "expected_direction": expected,
                "observed_direction": observed,
                "failed_directionality_test": bool(observed != 0 and observed != expected),
            }
        )
    return pd.DataFrame(tests).sort_values("mean_absolute_risk_change")
