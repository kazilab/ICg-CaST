"""Feature-space counterfactual stress tests for baseline models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def counterfactual_tests(model, test_df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Apply mechanism-specific feature perturbations and summarize risk changes."""
    base_x = test_df[cols].copy()
    base_pred = model.predict_proba(base_x)[:, 1]
    tests = []

    interventions = {
        "do_DNA_repair_rescue": {
            "scale_contains": [
                "DNA_adduct",
                "sig_activity_SBS4_like",
                "sig_activity_SBS24_like",
                "sig_activity_SBS22_like",
                "mut_total_count",
            ],
            "factor": 0.55,
            "expected_direction": -1,
        },
        "do_ROS_inflammation_blockade": {
            "scale_contains": ["ROS", "oxidative", "inflammation", "inflammatory", "NFkB", "SASP"],
            "factor": 0.50,
            "expected_direction": -1,
        },
        "do_epigenetic_memory_reset": {
            "scale_contains": ["epi_", "epigenetic_age", "PRC2", "histone", "chromatin", "methyl"],
            "factor": 0.45,
            "expected_direction": -1,
        },
        "do_proliferation_suppression": {
            "scale_contains": ["proliferation", "cell_cycle", "replicative", "nutrient", "clone_fraction"],
            "factor": 0.50,
            "expected_direction": -1,
        },
        "do_immune_surveillance_restore": {
            "scale_contains": ["immune_evasion", "immunosuppression"],
            "factor": 0.35,
            "increase_contains": ["immune_clearance", "host_immune_surveillance"],
            "increase": 1.25,
            "expected_direction": -1,
        },
    }

    for name, spec in interventions.items():
        x_cf = base_x.copy()
        for col in x_cf.columns:
            if any(token in col for token in spec.get("scale_contains", [])):
                x_cf[col] = x_cf[col] * spec["factor"]
            if any(token in col for token in spec.get("increase_contains", [])):
                x_cf[col] = x_cf[col] * spec["increase"]
        cf_pred = model.predict_proba(x_cf)[:, 1]
        delta = cf_pred - base_pred
        expected = int(spec["expected_direction"])
        observed = int(np.sign(float(np.mean(delta))))
        tests.append(
            {
                "intervention": name,
                "mean_predicted_risk_before": float(np.mean(base_pred)),
                "mean_predicted_risk_after": float(np.mean(cf_pred)),
                "mean_absolute_risk_change": float(np.mean(delta)),
                "median_absolute_risk_change": float(np.median(delta)),
                "expected_direction": expected,
                "observed_direction": observed,
                "failed_directionality_test": bool(observed != 0 and observed != expected),
            }
        )
    return pd.DataFrame(tests).sort_values("mean_absolute_risk_change")
