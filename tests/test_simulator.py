from __future__ import annotations

import numpy as np
import pandas as pd

from icg_cast import KCC_NAMES, SimConfig, simulate_cohort


def test_simulate_cohort_is_deterministic() -> None:
    cfg = SimConfig(n=20, months=6, seed=1)
    first, _ = simulate_cohort(cfg)
    second, _ = simulate_cohort(cfg)

    pd.testing.assert_frame_equal(first, second)


def test_simulate_cohort_schema_and_bounds() -> None:
    cohort, _ = simulate_cohort(SimConfig(n=20, months=6, seed=1))
    kcc_cols = [c for c in cohort.columns if c.startswith("kcc")]

    assert len(cohort) == 20
    assert len(KCC_NAMES) == 10
    assert len(kcc_cols) == 10
    assert cohort["coefficient_seed"].eq(-1).all()
    assert cohort.select_dtypes("number").replace([np.inf, -np.inf], np.nan).notna().all().all()
    assert cohort["state_final_clone_fraction"].between(0, 1).all()
    assert cohort["state_final_latent_risk"].between(0, 1).all()


def test_prior_sample_mode_is_deterministic_and_non_degenerate() -> None:
    cfg = SimConfig(
        n=30,
        months=8,
        seed=2,
        coefficient_mode="prior_sample",
        coefficient_seed=77,
    )
    first, _ = simulate_cohort(cfg)
    second, _ = simulate_cohort(cfg)
    pd.testing.assert_frame_equal(first, second)
    assert first["coefficient_seed"].eq(77).all()
    assert first.select_dtypes("number").replace([np.inf, -np.inf], np.nan).notna().all().all()
    assert first["state_final_clone_fraction"].between(0, 1).all()
    assert first["state_final_latent_risk"].between(0, 1).all()

    other, _ = simulate_cohort(
        SimConfig(
            n=30,
            months=8,
            seed=2,
            coefficient_mode="prior_sample",
            coefficient_seed=78,
        )
    )
    assert not first["future_event_probability"].equals(other["future_event_probability"])


def test_prior_sample_interval_brackets_point_mode_for_one_archetype() -> None:
    cfg_kwargs = {
        "n": 90,
        "months": 36,
        "seed": 7,
        "archetype_prior": {"pah_tobacco_like": 1.0},
    }
    point, _ = simulate_cohort(SimConfig(**cfg_kwargs))
    point_rate = float(point["future_event_probability"].mean())

    sampled_rates = []
    for coefficient_seed in range(40, 56):
        cohort, _ = simulate_cohort(
            SimConfig(
                **cfg_kwargs,
                coefficient_mode="prior_sample",
                coefficient_seed=coefficient_seed,
            )
        )
        sampled_rates.append(float(cohort["future_event_probability"].mean()))

    low, high = np.percentile(sampled_rates, [5, 95])
    assert high - low > 0.001
    assert low <= point_rate <= high
