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
from typing import Literal

import numpy as np
import pandas as pd

SurvivalReason = Literal["event", "administrative", "truncated"]


def time_to_event(
    trajectory: pd.DataFrame,
    column: str = "latent_risk",
    threshold: float = 0.5,
    horizon: int | None = None,
) -> tuple[int, int, SurvivalReason]:
    """Return ``(time_index, event_observed, reason)`` for threshold crossing.

    Right-censors at `horizon` (or the length of the trajectory if omitted).
    Time is reported as the 1-indexed month of the first crossing, matching
    the existing simulator schema. ``reason`` is ``"event"`` for an observed
    crossing, ``"administrative"`` for censoring at the requested horizon, and
    ``"truncated"`` when the trajectory ends before the requested horizon.
    """
    if column not in trajectory.columns:
        raise KeyError(f"trajectory missing column: {column}")
    if horizon is not None and (not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0):
        raise ValueError("horizon must be a positive integer or None")
    values = trajectory[column].to_numpy()
    observed_months = len(values)
    analysis_months = observed_months if horizon is None else min(horizon, observed_months)
    crossings = np.where(values[:analysis_months] >= threshold)[0]
    if crossings.size == 0:
        if horizon is not None and observed_months < horizon:
            return int(observed_months), 0, "truncated"
        return int(analysis_months), 0, "administrative"
    return int(crossings[0] + 1), 1, "event"


def add_survival_columns(
    cohort: pd.DataFrame,
    trajectories: Mapping[str, pd.DataFrame],
    horizon: int,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Append `time_to_high_risk_threshold` and `event_observed` to a cohort.

    `trajectories` must map every cohort `sample_id` to its per-month trajectory
    dataframe. Missing trajectories raise instead of being silently censored.
    """
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if "sample_id" not in cohort.columns:
        raise KeyError("cohort missing column: sample_id")
    out = cohort.copy()
    times = np.full(len(out), fill_value=horizon, dtype=int)
    events = np.zeros(len(out), dtype=int)
    reasons: list[SurvivalReason] = ["administrative"] * len(out)
    missing = [str(sid) for sid in out["sample_id"].tolist() if sid not in trajectories]
    if missing:
        raise KeyError(f"missing trajectories for sample_id(s): {missing}")
    for i, sid in enumerate(out["sample_id"].tolist()):
        t, e, reason = time_to_event(trajectories[sid], threshold=threshold, horizon=horizon)
        times[i] = t
        events[i] = e
        reasons[i] = reason
    out["time_to_high_risk_threshold"] = times
    out["event_observed"] = events
    out["time_to_event_reason"] = reasons
    return out


def restricted_mean_survival(times: np.ndarray, events: np.ndarray, horizon: int) -> float:
    """Kaplan-Meier-based RMST up to `horizon`.

    Uses a grouped step-function integral. Events after `horizon` are treated
    as administratively censored at `horizon`. For tied rows at the same time,
    events are applied in one canonical Kaplan-Meier step before same-time
    censoring removes subjects from the risk set.
    """
    times = np.asarray(times, dtype=int)
    events = np.asarray(events, dtype=int)
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 0:
        raise ValueError("horizon must be a non-negative integer")
    if times.size != events.size:
        raise ValueError("times and events must have the same length")
    if not np.isin(events, [0, 1]).all():
        raise ValueError("events must be encoded as 0/1")
    if times.size == 0:
        return float(horizon)

    n_at_risk = times.size
    surv = 1.0
    rmst = 0.0
    prev_t = 0

    for t in sorted(int(t) for t in np.unique(times[times <= horizon])):
        if t < prev_t:
            continue
        rmst += surv * (t - prev_t)
        at_time = times == t
        event_count = int(np.sum(at_time & (events == 1)))
        if event_count and n_at_risk > 0:
            surv *= (n_at_risk - event_count) / n_at_risk
        n_at_risk -= int(np.sum(at_time))
        prev_t = t

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
    crosses `threshold`. The input must already be a per-month panel with
    ``sample_id`` and ``month`` columns; standard ``simulate_cohort`` output
    must be joined/melted with retained trajectories before calling this.
    This is a coarse proxy; the proper version would re-simulate.

    Returns (point_estimate, ci_low, ci_high) at 95%.
    """
    required = {"sample_id", "month"}
    missing = required - set(cohort.columns)
    if missing:
        raise KeyError(
            "counterfactual RMST requires a per-month panel with columns: "
            f"{sorted(required)}; missing {sorted(missing)}"
        )

    rng = np.random.default_rng(random_state)

    def _rmst_for(df: pd.DataFrame) -> float:
        df_reset = df.reset_index(drop=True)
        if df_reset.empty:
            return restricted_mean_survival(np.array([], dtype=int), np.array([], dtype=int), horizon)
        group_column = "_survival_id" if "_survival_id" in df_reset.columns else "sample_id"
        grouped = df_reset.groupby(group_column, sort=False)
        group_order = list(grouped.groups)
        model_frame = df_reset.drop(columns=["_survival_id"], errors="ignore")
        proba = model.predict_proba(model_frame)[:, 1]
        times = np.full(len(group_order), horizon, dtype=int)
        events = np.zeros_like(times)
        for i, group_id in enumerate(group_order):
            g = grouped.get_group(group_id)
            risk = proba[g.index.to_numpy()]
            crossings = np.where(risk >= threshold)[0]
            if crossings.size:
                times[i] = int(g["month"].to_numpy()[crossings[0]])
                events[i] = 1
        return restricted_mean_survival(times, events, horizon)

    def _bootstrap_panel(sample_ids: np.ndarray) -> pd.DataFrame:
        pieces = []
        for replicate_id, sid in enumerate(sample_ids):
            piece = cohort.loc[cohort["sample_id"] == sid].copy()
            piece["_survival_id"] = f"{sid}__bootstrap_{replicate_id}"
            pieces.append(piece)
        return pd.concat(pieces, ignore_index=True)

    base = _rmst_for(cohort)
    cf = _rmst_for(intervention(cohort))
    delta = float(cf - base)

    n = cohort["sample_id"].nunique()
    ids = cohort["sample_id"].unique()
    diffs = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        sample_ids = rng.choice(ids, size=n, replace=True)
        sub = _bootstrap_panel(sample_ids)
        diffs[b] = _rmst_for(intervention(sub)) - _rmst_for(sub)
    ci_low, ci_high = np.quantile(diffs, [0.025, 0.975])
    return delta, float(ci_low), float(ci_high)


def survival_table(
    cohort: pd.DataFrame,
    group_col: str = "chemical_archetype",
    horizon: int | None = None,
) -> pd.DataFrame:
    """Per-group RMST summary table for reporting and tests.

    When ``time_to_event_reason`` is present (as added by
    :func:`add_survival_columns`), the table also breaks the non-event count
    into administrative (``n_censored_administrative``, reached horizon
    without crossing the threshold) and truncated (``n_censored_truncated``,
    trajectory ended before horizon) bins so mixed-horizon analyses can tell
    the two apart.
    """
    if horizon is None:
        horizon = int(cohort["time_to_high_risk_threshold"].max())
    has_reason = "time_to_event_reason" in cohort.columns
    rows: list[dict[str, object]] = []
    for name, sub in cohort.groupby(group_col):
        rmst = restricted_mean_survival(
            sub["time_to_high_risk_threshold"].to_numpy(),
            sub["event_observed"].to_numpy(),
            horizon=horizon,
        )
        row: dict[str, object] = {
            group_col: name,
            "n": int(len(sub)),
            "event_rate": float(sub["event_observed"].mean()),
            "rmst": rmst,
            "horizon": int(horizon),
        }
        if has_reason:
            reasons = sub["time_to_event_reason"].astype(str)
            row["n_events"] = int((reasons == "event").sum())
            row["n_censored_administrative"] = int((reasons == "administrative").sum())
            row["n_censored_truncated"] = int((reasons == "truncated").sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("rmst", ascending=False).reset_index(drop=True)
