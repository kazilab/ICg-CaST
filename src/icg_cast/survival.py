"""Time-to-event outcomes and causal time-shift estimands.

PLAN.md reference: section 7.6. Replaces the binary
`future_cancer_transition_event` with a time-to-threshold survival outcome
and supports counterfactual RMST (Restricted Mean Survival Time) shifts under
do-interventions.

This module deliberately avoids `lifelines`, `pysurvival`, and other extra
dependencies for v0.1 because the survival task is simple right-censored RMST.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd


def time_to_event(
    trajectory: pd.DataFrame,
    column: str = "latent_risk",
    threshold: float = 0.5,
    horizon: int | None = None,
) -> tuple[int, int]:
    """Return (time_index, event_observed) for the first crossing of `threshold`.

    Right-censors at `horizon` (or the length of the trajectory if omitted).
    Time is reported as the 1-indexed month of the first crossing, matching
    the existing simulator schema.
    """
    if column not in trajectory.columns:
        raise KeyError(f"trajectory missing column: {column}")
    values = trajectory[column].to_numpy()
    n = len(values) if horizon is None else min(horizon, len(values))
    crossings = np.where(values[:n] >= threshold)[0]
    if crossings.size == 0:
        return int(n), 0
    return int(crossings[0] + 1), 1


def add_survival_columns(
    cohort: pd.DataFrame,
    trajectories: Mapping[str, pd.DataFrame],
    threshold: float = 0.5,
    horizon: int | None = None,
) -> pd.DataFrame:
    """Append `time_to_high_risk_threshold` and `event_observed` to a cohort.

    `trajectories` maps `sample_id` to the per-month trajectory dataframe. Rows
    in `cohort` without a recorded trajectory receive censored entries at the
    horizon.
    """
    out = cohort.copy()
    times = np.full(len(out), fill_value=horizon if horizon is not None else 0, dtype=int)
    events = np.zeros(len(out), dtype=int)
    for i, sid in enumerate(out["sample_id"].tolist()):
        traj = trajectories.get(sid)
        if traj is None:
            continue
        t, e = time_to_event(traj, threshold=threshold, horizon=horizon)
        times[i] = t
        events[i] = e
    out["time_to_high_risk_threshold"] = times
    out["event_observed"] = events
    return out


def restricted_mean_survival(times: np.ndarray, events: np.ndarray, horizon: int) -> float:
    """Kaplan-Meier-based RMST up to `horizon`.

    Uses the standard step-function integral. For small synthetic cohorts this
    is faster and dependency-free; it is not intended for analyses with many
    tied events or interval censoring.
    """
    times = np.asarray(times, dtype=int)
    events = np.asarray(events, dtype=int)
    if times.size == 0:
        return float(horizon)

    order = np.argsort(times)
    t_sorted = times[order]
    e_sorted = events[order]
    n_at_risk = times.size
    surv = 1.0
    rmst = 0.0
    prev_t = 0

    for t, e in zip(t_sorted, e_sorted, strict=False):
        t_clip = int(min(t, horizon))
        rmst += surv * (t_clip - prev_t)
        if e == 1 and n_at_risk > 0:
            surv *= (n_at_risk - 1) / n_at_risk
        n_at_risk -= 1
        prev_t = t_clip
        if t_clip >= horizon:
            break

    if prev_t < horizon:
        rmst += surv * (horizon - prev_t)
    return float(rmst)


def counterfactual_rmst_difference(
    model,
    cohort: pd.DataFrame,
    intervention: Callable[[pd.DataFrame], pd.DataFrame],
    horizon: int,
    threshold: float = 0.5,
    n_bootstrap: int = 200,
    random_state: int | None = 7,
) -> tuple[float, float, float]:
    """Counterfactual RMST difference under a feature-space intervention.

    The intervention is applied via a user-supplied callable that takes a
    feature dataframe and returns the perturbed one. Because we do not re-run
    the simulator, the time-to-event under intervention is inferred from the
    model's risk trajectory: months are counted until predicted risk first
    crosses `threshold`. This is a coarse proxy and should be documented as
    such; the proper version would re-simulate.

    Returns (point_estimate, ci_low, ci_high) at 95%.
    """
    if "month" not in cohort.columns:
        raise KeyError("counterfactual RMST requires a per-month cohort layout (month column)")

    rng = np.random.default_rng(random_state)

    def _rmst_for(df: pd.DataFrame) -> float:
        proba = model.predict_proba(df)[:, 1]
        times = np.full(df["sample_id"].nunique(), horizon, dtype=int)
        events = np.zeros_like(times)
        for i, (_, g) in enumerate(df.groupby("sample_id", sort=False)):
            risk = proba[g.index.to_numpy()]
            crossings = np.where(risk >= threshold)[0]
            if crossings.size:
                times[i] = int(g["month"].to_numpy()[crossings[0]])
                events[i] = 1
        return restricted_mean_survival(times, events, horizon)

    base = _rmst_for(cohort)
    cf = _rmst_for(intervention(cohort))
    delta = float(cf - base)

    n = cohort["sample_id"].nunique()
    ids = cohort["sample_id"].unique()
    diffs = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        sample_ids = rng.choice(ids, size=n, replace=True)
        sub = cohort[cohort["sample_id"].isin(sample_ids)]
        diffs[b] = _rmst_for(intervention(sub)) - _rmst_for(sub)
    ci_low, ci_high = np.quantile(diffs, [0.025, 0.975])
    return delta, float(ci_low), float(ci_high)


def survival_table(
    cohort: pd.DataFrame,
    group_col: str = "chemical_archetype",
    horizon: int | None = None,
) -> pd.DataFrame:
    """Per-group RMST summary table for reporting and tests."""
    if horizon is None:
        horizon = int(cohort["time_to_high_risk_threshold"].max())
    rows: list[dict[str, object]] = []
    for name, sub in cohort.groupby(group_col):
        rmst = restricted_mean_survival(
            sub["time_to_high_risk_threshold"].to_numpy(),
            sub["event_observed"].to_numpy(),
            horizon=horizon,
        )
        rows.append({
            group_col: name,
            "n": int(len(sub)),
            "event_rate": float(sub["event_observed"].mean()),
            "rmst": rmst,
            "horizon": int(horizon),
        })
    return pd.DataFrame(rows).sort_values("rmst", ascending=False).reset_index(drop=True)
