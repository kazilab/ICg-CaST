from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from icg_cast import restricted_mean_survival, time_to_event
from icg_cast.survival import add_survival_columns, counterfactual_rmst_difference


def test_time_to_event_crossing_and_censoring() -> None:
    crossed = pd.DataFrame({"latent_risk": [0.1, 0.3, 0.55, 0.7]})
    censored = pd.DataFrame({"latent_risk": [0.1, 0.3, 0.49, 0.4]})
    truncated = pd.DataFrame({"latent_risk": [0.1, 0.3]})

    assert time_to_event(crossed, threshold=0.5) == (3, 1, "event")
    assert time_to_event(censored, threshold=0.5) == (4, 0, "administrative")
    assert time_to_event(crossed, threshold=0.5, horizon=2) == (2, 0, "administrative")
    assert time_to_event(truncated, threshold=0.5, horizon=4) == (2, 0, "truncated")


def test_add_survival_columns_requires_complete_trajectories_and_horizon() -> None:
    cohort = pd.DataFrame({"sample_id": ["S0", "S1"], "chemical_archetype": ["a", "b"]})
    trajectories = {
        "S0": pd.DataFrame({"latent_risk": [0.1, 0.6, 0.7]}),
        "S1": pd.DataFrame({"latent_risk": [0.1, 0.2, 0.3]}),
    }

    out = add_survival_columns(cohort, trajectories, horizon=3, threshold=0.5)

    assert out["time_to_high_risk_threshold"].tolist() == [2, 3]
    assert out["event_observed"].tolist() == [1, 0]
    assert out["time_to_event_reason"].tolist() == ["event", "administrative"]

    with pytest.raises(KeyError, match="missing trajectories"):
        add_survival_columns(cohort, {"S0": trajectories["S0"]}, horizon=3)


def test_restricted_mean_survival_is_finite() -> None:
    rmst = restricted_mean_survival(
        times=np.array([2, 4, 6]),
        events=np.array([1, 0, 1]),
        horizon=6,
    )

    assert np.isfinite(rmst)
    assert 0 <= rmst <= 6


def test_restricted_mean_survival_groups_ties_and_censors_after_horizon() -> None:
    tied = restricted_mean_survival(
        times=np.array([2, 2]),
        events=np.array([0, 1]),
        horizon=4,
    )
    post_horizon = restricted_mean_survival(
        times=np.array([2, 10]),
        events=np.array([1, 1]),
        horizon=4,
    )
    censored_after_horizon = restricted_mean_survival(
        times=np.array([2, 10]),
        events=np.array([1, 0]),
        horizon=4,
    )

    assert tied == pytest.approx(3.0)
    assert post_horizon == pytest.approx(censored_after_horizon)


def test_counterfactual_rmst_difference_bootstrap_handles_resampled_indices() -> None:
    """The bootstrap loop must work on subsamples whose row labels are not 0..N-1.

    Regression test: the previous implementation indexed ``proba`` with the
    original DataFrame labels, which raised ``IndexError`` on any subsample.
    """
    rng = np.random.default_rng(0)
    records = []
    for sid in range(5):
        for month in range(1, 5):
            records.append(
                {
                    "sample_id": f"S{sid}",
                    "month": month,
                    "feature_a": rng.normal(),
                    "feature_b": rng.normal(),
                }
            )
    cohort = pd.DataFrame.from_records(records)

    class _DeterministicModel:
        def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
            n = len(df)
            risk = np.full(n, 0.1)
            mask = (df["sample_id"].to_numpy() == "S2") & (df["month"].to_numpy() >= 3)
            risk[mask] = 0.9
            return np.column_stack([1 - risk, risk])

    delta, ci_low, ci_high = counterfactual_rmst_difference(
        model=_DeterministicModel(),
        cohort=cohort,
        intervention=lambda df: df.copy(),
        horizon=4,
        threshold=0.5,
        n_bootstrap=10,
        random_state=0,
    )

    assert np.isfinite(delta)
    assert np.isfinite(ci_low) and np.isfinite(ci_high)
    assert ci_low <= ci_high


def test_counterfactual_rmst_difference_bootstrap_preserves_duplicate_draws() -> None:
    records = []
    for sid in range(3):
        for month in range(1, 4):
            records.append({"sample_id": f"S{sid}", "month": month, "feature": float(month)})
    cohort = pd.DataFrame.from_records(records)
    seen_duplicate_draw = []

    class _FlatModel:
        def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
            risk = np.where(df["month"].to_numpy() >= 2, 0.6, 0.1)
            return np.column_stack([1 - risk, risk])

    def intervention(df: pd.DataFrame) -> pd.DataFrame:
        if "_survival_id" in df.columns:
            seen_duplicate_draw.append(df["_survival_id"].nunique() > df["sample_id"].nunique())
        return df.copy()

    counterfactual_rmst_difference(
        model=_FlatModel(),
        cohort=cohort,
        intervention=intervention,
        horizon=3,
        threshold=0.5,
        n_bootstrap=25,
        random_state=0,
    )

    assert any(seen_duplicate_draw)
