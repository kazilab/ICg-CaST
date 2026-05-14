from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from icg_cast import SimConfig, simulate_cohort
from icg_cast.models import (
    biological_coherence_summary,
    evaluate_bundle,
    feature_sets,
    train_baselines,
    validate_no_target_leakage,
)


@pytest.fixture(scope="module")
def trained_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    cohort, _ = simulate_cohort(SimConfig(n=80, months=72, seed=7))
    metrics, importance, counterfactual, bundle = train_baselines(cohort, seed=7)
    return cohort, metrics, importance, counterfactual, bundle


def test_feature_sets_exclude_target_derived_columns(trained_outputs) -> None:
    cohort = trained_outputs[0]
    forbidden = {
        "future_cancer_transition_event",
        "future_event_probability",
        "high_risk_transition_state",
        "state_final_latent_risk",
        "state_auc_latent_risk",
    }

    for set_name, cols in feature_sets(cohort).items():
        assert forbidden.isdisjoint(cols), set_name
        validate_no_target_leakage(cols, feature_set=set_name)


@pytest.mark.parametrize(
    "column",
    [
        "future_cancer_transition_event",
        "future_event_probability",
        "high_risk_transition_state",
        "state_final_latent_risk",
        "state_auc_latent_risk",
    ],
)
def test_target_leakage_guard_rejects_endpoint_columns(column: str) -> None:
    with pytest.raises(ValueError, match="target-derived"):
        validate_no_target_leakage(["dose", column])


def test_train_baselines_output_schema(trained_outputs) -> None:
    _cohort, metrics, importance, counterfactual, bundle = trained_outputs

    assert {"feature_set", "model", "n_features", "roc_auc", "average_precision", "brier_score"}.issubset(
        metrics.columns
    )
    assert {"logistic_l2", "random_forest", "extra_trees"}.issubset(set(metrics["model"]))
    assert "multiomics_plus_qAOP" in set(metrics["feature_set"])
    assert metrics[["roc_auc", "average_precision", "brier_score"]].apply(np.isfinite).all().all()

    assert {
        "feature",
        "permutation_importance_mean_auc_drop",
        "permutation_importance_sd",
        "best_multiomics_model",
    }.issubset(importance.columns)
    assert len(importance) == len(bundle["feature_columns"])
    assert bundle["target"] == "future_cancer_transition_event"
    validate_no_target_leakage(bundle["feature_columns"], target=str(bundle["target"]))

    assert {
        "intervention",
        "expected_direction",
        "observed_direction",
        "failed_directionality_test",
        "mean_absolute_risk_change",
    }.issubset(counterfactual.columns)


def test_evaluate_bundle_output_schema(trained_outputs) -> None:
    cohort, _metrics, _importance, _counterfactual, bundle = trained_outputs
    metrics, calibration, counterfactual, coherence = evaluate_bundle(cohort, bundle)

    assert {"roc_auc", "average_precision", "brier_score", "event_rate", "mean_predicted_risk", "n"}.issubset(
        metrics.columns
    )
    assert metrics[["roc_auc", "average_precision", "brier_score"]].apply(np.isfinite).all().all()

    assert {
        "bin",
        "bin_low",
        "bin_high",
        "n",
        "observed_event_rate",
        "mean_predicted_risk",
        "expected_calibration_error",
    }.issubset(calibration.columns)
    assert "summary" in set(calibration["bin"].astype(str))

    assert {"intervention", "expected_direction", "observed_direction", "failed_directionality_test"}.issubset(
        counterfactual.columns
    )
    assert counterfactual["failed_directionality_test"].map(type).eq(bool).all()

    assert {"tested_intervention_count", "correct_direction_count", "biological_coherence_score"}.issubset(
        coherence.columns
    )
    score = float(coherence["biological_coherence_score"].iloc[0])
    assert 0.0 <= score <= 1.0


def test_biological_coherence_summary_scores_directionality() -> None:
    counterfactual = pd.DataFrame(
        [
            {"expected_direction": -1, "observed_direction": -1},
            {"expected_direction": -1, "observed_direction": 1},
            {"expected_direction": 0, "observed_direction": 1},
        ]
    )

    summary = biological_coherence_summary(counterfactual)

    assert summary["tested_intervention_count"].iloc[0] == 2
    assert summary["correct_direction_count"].iloc[0] == 1
    assert summary["biological_coherence_score"].iloc[0] == pytest.approx(0.5)
