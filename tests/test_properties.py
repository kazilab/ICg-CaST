from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from icg_cast import SimConfig, simulate_cohort
from icg_cast.biology.biological_risk_equation import biological_risk_equation
from icg_cast.constants import STATE_NAMES
from icg_cast.oracle.reference_risk_oracle import reference_risk_oracle
from icg_cast.simulator import simulate_state_trajectory

try:
    from hypothesis import given, settings, strategies as st  # noqa: I001
except ImportError:
    HAS_HYPOTHESIS = False
else:
    HAS_HYPOTHESIS = True


if not HAS_HYPOTHESIS:

    def test_property_tests_require_hypothesis() -> None:
        pytest.skip("install the dev extra to run Hypothesis property tests")


else:

    @st.composite
    def _sim_configs(draw) -> SimConfig:
        return SimConfig(
            n=draw(st.integers(min_value=5, max_value=25)),
            months=draw(st.integers(min_value=2, max_value=18)),
            seed=draw(st.integers(min_value=0, max_value=10_000)),
            simulator_backend=draw(st.sampled_from(["python", "vectorized"])),
        )

    def _assert_bounded_finite(series: pd.Series, *, low: float, high: float) -> None:
        values = series.to_numpy(dtype=float)
        assert np.isfinite(values).all()
        assert (values >= low).all()
        assert (values <= high).all()

    @settings(max_examples=16, deadline=None)
    @given(cfg=_sim_configs())
    def test_simulated_cohort_state_invariants_hold(cfg: SimConfig) -> None:
        cohort, trajectories = simulate_cohort(cfg)

        assert len(cohort) == cfg.n
        assert {"low", "median", "high"} <= set(trajectories)
        assert cohort["high_risk_transition_state"].isin([0, 1]).all()
        _assert_bounded_finite(cohort["future_event_probability"], low=0.0, high=1.0)

        for name in STATE_NAMES:
            _assert_bounded_finite(cohort[f"state_final_{name}"], low=0.0, high=np.inf)
            _assert_bounded_finite(cohort[f"state_auc_{name}"], low=0.0, high=np.inf)

        for name in ("proliferation", "clone_fraction", "latent_risk"):
            _assert_bounded_finite(cohort[f"state_final_{name}"], low=0.0, high=1.0)
            _assert_bounded_finite(cohort[f"state_auc_{name}"], low=0.0, high=1.0)

    @settings(max_examples=20, deadline=None)
    @given(
        kcc=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=10,
            max_size=10,
        ),
        dose=st.floats(min_value=0.0, max_value=6.0, allow_nan=False, allow_infinity=False),
        months=st.integers(min_value=1, max_value=24),
        seed=st.integers(min_value=0, max_value=10_000),
    )
    def test_state_trajectory_state_invariants_hold(
        kcc: list[float], dose: float, months: int, seed: int
    ) -> None:
        susceptibility = {
            "repair_capacity": 0.9,
            "antioxidant_capacity": 0.9,
            "immune_surveillance": 0.9,
            "detox_balance": 1.0,
            "baseline_proliferation": 0.0,
        }
        trajectory = simulate_state_trajectory(
            np.asarray(kcc, dtype=float),
            dose=dose,
            susceptibility=susceptibility,
            months=months,
            rng=np.random.default_rng(seed),
        )

        assert len(trajectory) == months
        assert trajectory["month"].tolist() == list(range(1, months + 1))
        for name in STATE_NAMES:
            _assert_bounded_finite(trajectory[name], low=0.0, high=np.inf)
        for name in ("proliferation", "clone_fraction", "latent_risk"):
            _assert_bounded_finite(trajectory[name], low=0.0, high=1.0)

    @settings(max_examples=20, deadline=None)
    @given(
        base=st.fixed_dictionaries(
            {
                "DNA_adducts": st.floats(min_value=0.01, max_value=8.0),
                "ROS": st.floats(min_value=0.01, max_value=8.0),
                "inflammation": st.floats(min_value=0.01, max_value=8.0),
                "epigenetic_age": st.floats(min_value=0.01, max_value=12.0),
                "proliferation": st.floats(min_value=0.01, max_value=0.95),
                "driver_count_proxy": st.floats(min_value=0.01, max_value=8.0),
                "clone_fraction": st.floats(min_value=0.01, max_value=0.95),
                "immune_clearance": st.floats(min_value=0.1, max_value=1.1),
            }
        ),
        multiplier=st.floats(
            min_value=1.05, max_value=2.0, allow_nan=False, allow_infinity=False
        ),
    )
    def test_latent_risk_monotonicity_for_structural_inputs(
        base: dict[str, float], multiplier: float
    ) -> None:
        base_risk = float(reference_risk_oracle(base))
        bio_base_risk = float(biological_risk_equation(base))

        harmful_inputs = (
            "DNA_adducts",
            "ROS",
            "inflammation",
            "epigenetic_age",
            "proliferation",
            "driver_count_proxy",
            "clone_fraction",
        )
        for name in harmful_inputs:
            perturbed = dict(base)
            perturbed[name] *= multiplier
            assert float(reference_risk_oracle(perturbed)) > base_risk
            assert float(biological_risk_equation(perturbed)) > bio_base_risk

        protected = dict(base)
        protected["immune_clearance"] *= multiplier
        assert float(reference_risk_oracle(protected)) < base_risk
        assert float(biological_risk_equation(protected)) < bio_base_risk
