"""ICg-Bench task implementations.

Each task accepts a fitted model plus the cohort data it requires and returns
a flat dict of metrics. Tasks are deliberately model-agnostic: they only
require `predict_proba` and (for the latent-recovery task) `predict_bottleneck`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    r2_score,
    roc_auc_score,
)


class _SupportsProba(Protocol):
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


class _SupportsBottleneck(Protocol):
    def predict_bottleneck(self, X: pd.DataFrame) -> pd.DataFrame: ...
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...
    def intervene(self, unit: str, scale: float | None = None, shift: float | None = None) -> None: ...
    def clear_interventions(self) -> None: ...


def task_risk_prediction(
    model: _SupportsProba,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Predictive discrimination + calibration on held-out subjects."""
    proba = model.predict_proba(X_test)[:, 1]
    proba_clipped = np.clip(proba, 1e-6, 1.0 - 1e-6)
    return {
        "auroc": float(roc_auc_score(y_test, proba)),
        "auprc": float(average_precision_score(y_test, proba)),
        "brier": float(brier_score_loss(y_test, proba_clipped)),
        "mean_proba": float(np.mean(proba)),
        "event_rate": float(np.mean(y_test)),
    }


def task_latent_recovery(
    model: _SupportsBottleneck,
    X_test: pd.DataFrame,
    S_true: pd.DataFrame,
) -> dict[str, float]:
    """Per-state R^2 of the bottleneck against true qAOP state."""
    S_hat = model.predict_bottleneck(X_test)
    common = [c for c in S_true.columns if c in S_hat.columns]
    if not common:
        raise ValueError("S_true and S_hat share no columns")
    per_state = {
        f"r2__{c}": float(r2_score(S_true[c].to_numpy(), S_hat[c].to_numpy()))
        for c in common
    }
    per_state["r2_mean"] = float(np.mean(list(per_state.values())))
    per_state["n_states"] = int(len(common))
    return per_state


def task_intervention_conformity(
    model: _SupportsBottleneck,
    X_test: pd.DataFrame,
    interventions: Mapping[str, Mapping[str, float]],
    expected_directions: Mapping[str, int],
    tolerance: float = 1e-3,
) -> dict[str, float]:
    """Mechanism counterfactual conformity for bottleneck-aware models.

    `interventions` maps intervention name -> {bottleneck_unit: scale_factor}.
    `expected_directions` maps intervention name -> -1 / 0 / +1.
    """
    base = model.predict_proba(X_test)[:, 1]
    passes = 0
    scored = 0
    deltas: dict[str, float] = {}
    for name, spec in interventions.items():
        model.clear_interventions()
        for unit, scale in spec.items():
            model.intervene(unit=unit, scale=float(scale))
        after = model.predict_proba(X_test)[:, 1]
        delta = float(np.mean(after - base))
        deltas[name] = delta
        expected = int(expected_directions.get(name, 0))
        if expected == 0:
            continue
        scored += 1
        if np.sign(delta) == np.sign(expected) or abs(delta) <= tolerance:
            passes += 1
    model.clear_interventions()
    score = float(passes / scored) if scored else float("nan")
    out: dict[str, float] = {"conformity_score": score, "n_scored": float(scored)}
    out.update({f"delta__{k}": v for k, v in deltas.items()})
    return out


def task_cross_host_generalization(
    model: _SupportsProba,
    X_source_test: pd.DataFrame,
    y_source_test: np.ndarray,
    X_target_test: pd.DataFrame,
    y_target_test: np.ndarray,
) -> dict[str, float]:
    """Source vs. target AUROC under a host-distribution shift.

    The model is assumed to be already fitted on a *source* cohort. This task
    only evaluates; it never re-fits.
    """
    src_proba = model.predict_proba(X_source_test)[:, 1]
    tgt_proba = model.predict_proba(X_target_test)[:, 1]
    src_auc = float(roc_auc_score(y_source_test, src_proba))
    tgt_auc = float(roc_auc_score(y_target_test, tgt_proba))
    return {
        "auroc_source": src_auc,
        "auroc_target": tgt_auc,
        "transfer_gap": float(src_auc - tgt_auc),
    }
