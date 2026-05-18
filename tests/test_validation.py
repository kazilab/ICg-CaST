"""Tests for the validation subpackage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from icg_cast.validation import (
    biological_coherence_score,
    biological_coherence_summary,
    calibration_curve,
    effect_weighted_biological_coherence,
    expected_calibration_error,
    human_relevance_transfer_index,
    pathway_attribution_consistency,
    severity_effect_weighted_biological_coherence,
    severity_weighted_biological_coherence,
)


def test_biological_coherence_score_matches_expected_directions() -> None:
    counterfactual = pd.DataFrame(
        [
            {"intervention": "do_a", "expected_direction": -1, "observed_direction": -1},
            {"intervention": "do_b", "expected_direction": -1, "observed_direction": -1},
            {"intervention": "do_c", "expected_direction": +1, "observed_direction": -1},
            {"intervention": "do_d", "expected_direction": 0, "observed_direction": +1},
        ]
    )

    score = biological_coherence_score(counterfactual)

    summary = biological_coherence_summary(counterfactual)
    assert summary["tested_intervention_count"].iloc[0] == 3
    assert summary["correct_direction_count"].iloc[0] == 2
    assert score == pytest.approx(2 / 3)


def test_effect_weighted_biological_coherence_uses_risk_change_magnitude() -> None:
    counterfactual = pd.DataFrame(
        [
            {"expected_direction": -1, "observed_direction": -1, "mean_absolute_risk_change": -0.20},
            {"expected_direction": -1, "observed_direction": +1, "mean_absolute_risk_change": 0.05},
            {"expected_direction": +1, "observed_direction": +1, "mean_absolute_risk_change": 0.15},
        ]
    )

    score = effect_weighted_biological_coherence(counterfactual)

    assert score == pytest.approx((0.20 + 0.15) / (0.20 + 0.05 + 0.15))


def test_severity_weighted_biological_coherence_uses_intervention_weight() -> None:
    counterfactual = pd.DataFrame(
        [
            {
                "expected_direction": -1,
                "observed_direction": -1,
                "mean_absolute_risk_change": -0.20,
                "intervention_severity_weight": 1.0,
            },
            {
                "expected_direction": -1,
                "observed_direction": +1,
                "mean_absolute_risk_change": 0.05,
                "intervention_severity_weight": 6.0,
            },
            {
                "expected_direction": +1,
                "observed_direction": +1,
                "mean_absolute_risk_change": 0.15,
                "intervention_severity_weight": 3.0,
            },
        ]
    )

    severity_score = severity_weighted_biological_coherence(counterfactual)
    severity_effect_score = severity_effect_weighted_biological_coherence(counterfactual)

    assert severity_score == pytest.approx((1.0 + 3.0) / (1.0 + 6.0 + 3.0))
    assert severity_effect_score == pytest.approx(
        (1.0 * 0.20 + 3.0 * 0.15)
        / (1.0 * 0.20 + 6.0 * 0.05 + 3.0 * 0.15)
    )


def test_pathway_attribution_consistency_groups_by_pathway() -> None:
    importance = pd.DataFrame(
        [
            {"feature": "tx_TP53", "permutation_importance_mean_auc_drop": 0.10},
            {"feature": "tx_MDM2", "permutation_importance_mean_auc_drop": 0.06},
            {"feature": "epi_PRC2", "permutation_importance_mean_auc_drop": 0.04},
            {"feature": "sig_activity_SBS4_like", "permutation_importance_mean_auc_drop": 0.02},
            {"feature": "tx_OFFTARGET", "permutation_importance_mean_auc_drop": -0.03},
        ]
    )
    pathway_map = {
        "tx_TP53": "p53_checkpoint",
        "tx_MDM2": "p53_checkpoint",
        "epi_PRC2": "stemness_PRC2",
    }

    table = pathway_attribution_consistency(importance, pathway_map)

    by_pathway = table.set_index("pathway")
    assert by_pathway.loc["p53_checkpoint", "n_features"] == 2
    assert by_pathway.loc["p53_checkpoint", "total_importance"] == pytest.approx(0.16)
    assert by_pathway.loc["unmapped", "total_importance_clipped"] == pytest.approx(0.0)
    # unmapped feature is preserved as its own pathway
    assert "unmapped" in by_pathway.index
    # shares sum to ~1
    assert table["share_of_total"].sum() == pytest.approx(1.0)


def test_expected_calibration_error_and_curve_on_perfectly_calibrated_data() -> None:
    rng = np.random.default_rng(0)
    proba = rng.uniform(size=2000)
    y = (rng.uniform(size=2000) < proba).astype(int)

    ece = expected_calibration_error(y, proba, n_bins=10)
    mean_pred, obs_frac, counts = calibration_curve(y, proba, n_bins=10)

    assert ece < 0.05
    assert len(mean_pred) == 10
    finite = np.isfinite(mean_pred)
    assert np.allclose(mean_pred[finite], obs_frac[finite], atol=0.10)
    assert counts.sum() == len(y)


def test_human_relevance_transfer_index_with_mixed_conservation() -> None:
    table = pd.DataFrame(
        [
            {"key_event": "DNA_adduct_formation", "conservation": "conserved",
             "human_activation": 0.9, "rodent_activation": 0.9},
            {"key_event": "p53_checkpoint", "conservation": "conserved",
             "human_activation": 0.8, "rodent_activation": 0.8},
            {"key_event": "alpha_2u_globulin_nephropathy", "conservation": "rodent_specific",
             "human_activation": 0.0, "rodent_activation": 0.9},
            {"key_event": "PPAR_alpha_proliferation", "conservation": "rodent_specific",
             "human_activation": 0.1, "rodent_activation": 0.8},
            {"key_event": "off_target_low_signal", "conservation": "conserved",
             "human_activation": 0.1, "rodent_activation": 0.1},
            {"key_event": "rodent_only_conserved_signal", "conservation": "conserved",
             "human_activation": 0.1, "rodent_activation": 0.9},
        ]
    )

    result = human_relevance_transfer_index(table)

    assert result.n_conserved_human == 2
    assert result.n_rodent_specific == 2
    assert result.n_conserved_inactive_in_human == 1
    assert result.score == pytest.approx(0.5)
    assert result.coverage_adjusted_score == pytest.approx(2 / 5)
    assert len(result.reasons) == len(table)


def test_human_relevance_transfer_index_is_nan_without_evidence() -> None:
    table = pd.DataFrame(
        [
            {"key_event": "ke1", "conservation": "conserved",
             "human_activation": 0.1, "rodent_activation": 0.9},
        ]
    )
    result = human_relevance_transfer_index(table)
    assert np.isnan(result.score)
    assert result.n_conserved_human == 0
    assert result.n_rodent_specific == 0
    assert result.n_conserved_inactive_in_human == 1
    assert result.coverage_adjusted_score == pytest.approx(0.0)
