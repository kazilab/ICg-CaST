from __future__ import annotations

import numpy as np
import pandas as pd

from icg_cast.benchmark.conformity_bootstrap import (
    bootstrap_conformity,
    prior_conformity_from_table,
    responsive_conformity_from_table,
    responsive_passes_from_table,
)


class _NoOpBottleneckModel:
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba = np.full(len(X), 0.5, dtype=float)
        return np.column_stack([1.0 - proba, proba])

    def clear_interventions(self) -> None:
        pass

    def intervene(self, unit: str, scale: float) -> None:
        pass


def test_prior_conformity_does_not_pass_zero_delta() -> None:
    score = prior_conformity_from_table(
        np.array([0.0, 0.01]),
        np.array([1, 1]),
        threshold=0.005,
    )

    assert score == 0.5


def test_responsive_threshold_adapts_to_delta_dispersion() -> None:
    passes = responsive_passes_from_table(
        np.array([0.01, 0.02, 0.20]),
        np.array([1, 1, 1]),
        threshold=0.005,
    )
    score = responsive_conformity_from_table(
        np.array([0.01, 0.02, 0.20]),
        np.array([1, 1, 1]),
        threshold=0.005,
    )

    assert passes.tolist() == [False, True, True]
    assert score == 2 / 3


def test_bootstrap_conformity_does_not_reward_noop_model() -> None:
    result = bootstrap_conformity(
        _NoOpBottleneckModel(),
        pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}),
        interventions={"raise": {"unit": 1.5}},
        expected_directions_prior={"raise": 1},
        expected_directions_dgp={"raise": 1},
        n_bootstrap=3,
        random_state=0,
        responsive_threshold=0.005,
    )

    assert result["prior_conformity"] == (0.0, 0.0, 0.0)
    assert result["dgp_conformity"] == (0.0, 0.0, 0.0)
    assert result["responsive_dgp_conformity"] == (0.0, 0.0, 0.0)
