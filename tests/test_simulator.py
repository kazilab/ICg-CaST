from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from icg_cast import KCC_NAMES, SimConfig, simulate_cohort, simulate_state_trajectory, summarize_trajectory
from icg_cast.constants import STATE_NAMES
from icg_cast.survival import add_survival_columns


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


def test_vectorized_backend_is_deterministic_and_schema_compatible() -> None:
    cfg = SimConfig(n=30, months=8, seed=5, simulator_backend="vectorized")
    first, first_traj = simulate_cohort(cfg)
    second, second_traj = simulate_cohort(cfg)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == cfg.n
    assert set(first_traj) == set(second_traj)
    assert first.select_dtypes("number").replace([np.inf, -np.inf], np.nan).notna().all().all()
    assert first["state_final_clone_fraction"].between(0, 1).all()
    assert first["state_final_latent_risk"].between(0, 1).all()
    assert all(len(traj) == cfg.months for traj in first_traj.values())


@pytest.mark.parametrize("backend", ["python", "vectorized"])
def test_event_probability_uses_cumulative_latent_risk(backend: str) -> None:
    cfg = SimConfig(n=8, months=5, seed=4, simulator_backend=backend, retain_trajectories=True)
    cohort, trajectories = simulate_cohort(cfg)
    sample_id = str(cohort["sample_id"].iloc[0])
    trajectory = trajectories[sample_id]

    # ``latent_risk`` is a per-month event probability (sigmoid output); the
    # cumulative hazard is Σ -log(1 - p_t), so the horizon event probability
    # is 1 - exp(-hazard_scale * cumulative_hazard) — equivalently
    # 1 - prod(1 - p_t) at hazard_scale = 1.
    per_month_hazard = -np.log1p(-np.clip(trajectory["latent_risk"].to_numpy(dtype=float), 0.0, 1.0 - 1e-12))
    expected_cumulative = per_month_hazard.sum()
    expected = 1.0 - np.exp(-cfg.event_hazard_scale * expected_cumulative)

    assert cohort["future_event_probability"].iloc[0] == pytest.approx(expected)
    assert cohort["state_cumulative_latent_risk"].iloc[0] == pytest.approx(expected_cumulative)


@pytest.mark.parametrize("backend", ["python", "vectorized"])
def test_one_month_auc_equals_single_observation(backend: str) -> None:
    cohort, _ = simulate_cohort(SimConfig(n=6, months=1, seed=6, simulator_backend=backend))

    for name in STATE_NAMES:
        assert np.allclose(cohort[f"state_auc_{name}"], cohort[f"state_final_{name}"])


def test_summarize_trajectory_one_month_auc_equals_value() -> None:
    rng = np.random.default_rng(0)
    susceptibility = {
        "repair_capacity": 0.8,
        "antioxidant_capacity": 0.8,
        "immune_surveillance": 0.8,
        "detox_balance": 1.0,
        "baseline_proliferation": 0.0,
    }
    trajectory = simulate_state_trajectory(
        np.zeros(len(KCC_NAMES)),
        1.0,
        susceptibility,
        1,
        rng,
    )
    states = summarize_trajectory(trajectory)

    for name in STATE_NAMES:
        assert states[f"state_auc_{name}"] == pytest.approx(states[f"state_final_{name}"])


@pytest.mark.parametrize("backend", ["python", "vectorized"])
def test_retain_trajectories_returns_all_samples_for_survival_consumers(backend: str) -> None:
    cfg = SimConfig(n=7, months=4, seed=3, simulator_backend=backend, retain_trajectories=True)
    cohort, trajectories = simulate_cohort(cfg)

    assert set(trajectories) == set(cohort["sample_id"])
    assert all(len(trajectory) == cfg.months for trajectory in trajectories.values())

    survival = add_survival_columns(cohort, trajectories, horizon=cfg.months)
    assert survival["time_to_high_risk_threshold"].between(1, cfg.months).all()
    assert survival["event_observed"].isin([0, 1]).all()


def test_sim_config_rejects_unknown_backend() -> None:
    cfg = SimConfig(n=5, months=3, simulator_backend="fast")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="simulator_backend"):
        cfg.validate()


def test_sim_config_rejects_bad_numeric_inputs() -> None:
    with pytest.raises(ValueError, match="n must be a positive integer"):
        SimConfig(n=0).validate()
    with pytest.raises(ValueError, match="dose_lognormal_sigma"):
        SimConfig(dose_lognormal_sigma=-0.1).validate()
    with pytest.raises(ValueError, match="archetype_prior weights"):
        SimConfig(archetype_prior={"inert_control": -1.0}).validate()


def test_simulate_state_trajectory_validates_public_inputs() -> None:
    rng = np.random.default_rng(0)
    susceptibility = {
        "repair_capacity": 0.8,
        "antioxidant_capacity": 0.8,
        "immune_surveillance": 0.8,
        "detox_balance": 1.0,
        "baseline_proliferation": 0.0,
    }

    with pytest.raises(ValueError, match="kcc values must be in \\[0, 1\\]"):
        simulate_state_trajectory(np.full(len(KCC_NAMES), 2.0), 1.0, susceptibility, 3, rng)
    with pytest.raises(ValueError, match="dose must be finite and non-negative"):
        simulate_state_trajectory(np.zeros(len(KCC_NAMES)), -1.0, susceptibility, 3, rng)
    with pytest.raises(ValueError, match="susceptibility\\['repair_capacity'\\]"):
        simulate_state_trajectory(
            np.zeros(len(KCC_NAMES)),
            1.0,
            {**susceptibility, "repair_capacity": 99.0},
            3,
            rng,
        )


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
