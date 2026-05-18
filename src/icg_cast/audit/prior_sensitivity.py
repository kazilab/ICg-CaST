"""Prior sensitivity audit for MB-CNet stage-2 sign constraints.

For each bottleneck unit, refits stage 2 with that coordinate *unconstrained*
(sign ``0``) while keeping the remaining signs at the structural prior. Reports
the change in responsive DGP conformity (and optionally AUROC on test) relative
to the fully constrained model.

This answers: "which elicited signs are *binding* for the counterfactual audit
on this cohort?"
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from icg_cast.benchmark.conformity_bootstrap import responsive_conformity_from_table
from icg_cast.bottleneck import (
    MechanismBottleneckClassifier,
    augment_with_interventions,
)


def _make_stage2_signs(mb: MechanismBottleneckClassifier, signs: list[int]):
    from icg_cast.bottleneck import _StandardisedSignConstrained

    return _StandardisedSignConstrained(
        signs=signs,
        class_weight="balanced",
        random_state=mb.random_state,
    )


@dataclass
class _MBStage2View:
    """Minimal view so :func:`MechanismBottleneckClassifier.score_intervention_conformity` works."""

    mb: MechanismBottleneckClassifier
    stage2: object

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.stage2.predict_proba(self.mb._bottleneck(X))

    def clear_interventions(self) -> None:
        self.mb.clear_interventions()

    def intervene(self, unit: str, scale: float | None = None, shift: float | None = None) -> None:
        self.mb.intervene(unit=unit, scale=scale, shift=shift)


def _responsive_conformity_mb(
    proxy,
    X_te: pd.DataFrame,
    interventions: Mapping[str, Mapping[str, float | int]],
    expected_directions_dgp: Mapping[str, int],
    *,
    threshold: float = 0.005,
) -> float:
    order = list(interventions.keys())
    base = proxy.predict_proba(X_te)[:, 1]
    means: list[float] = []
    exp_dgp: list[int] = []
    for name in order:
        proxy.clear_interventions()
        for unit, scale in interventions[name].items():
            proxy.intervene(unit=unit, scale=float(scale))
        after = proxy.predict_proba(X_te)[:, 1]
        means.append(float(np.mean(after - base)))
        exp_dgp.append(int(expected_directions_dgp[name]))
    proxy.clear_interventions()
    return responsive_conformity_from_table(
        np.asarray(means, dtype=float),
        np.asarray(exp_dgp, dtype=int),
        threshold=threshold,
    )


def _refit_stage2_relaxed(
    mb: MechanismBottleneckClassifier,
    X_train: pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    relaxed_signs: list[int],
) -> object:
    """Refit stage 2 only (same stage-1 prediction / augmentation policy as ``mb``)."""
    y_train = np.asarray(y_train, dtype=int).ravel()
    Xf = X_train[list(mb.feature_columns_)]
    S_hat = pd.DataFrame(
        mb.stage1_.predict(Xf.to_numpy()),
        columns=list(mb.bottleneck_units_),
    )

    stage2 = _make_stage2_signs(mb, relaxed_signs)

    if mb.stage2_kind == "sign_constrained_augmented":
        if mb.augment_interventions is None or (
            mb.augment_latent_risk_fn is None
            and getattr(mb, "augment_cumulative_risk_fn", None) is None
        ):
            raise ValueError("augmented MB-CNet missing augmentation config")
        rng = np.random.default_rng(mb.random_state)
        S_aug, y_aug, w_aug = augment_with_interventions(
            S_hat,
            interventions=mb.augment_interventions,
            latent_risk_fn=mb.augment_latent_risk_fn,
            cumulative_risk_fn=getattr(mb, "augment_cumulative_risk_fn", None),
            hazard_scale=mb.augment_hazard_scale,
            months=mb.augment_months,
            rng=rng,
            samples_per_intervention=mb.augment_samples_per_intervention,
        )
        S_combined = pd.concat([S_hat, S_aug], axis=0, ignore_index=True)
        y_combined = np.concatenate([y_train, y_aug])
        w_combined = np.concatenate([np.ones(len(y_train), dtype=float), w_aug])
        stage2.fit(S_combined, y_combined, sample_weight=w_combined)
    else:
        stage2.fit(S_hat, y_train)

    return stage2


def prior_sensitivity(
    mb: MechanismBottleneckClassifier,
    X_train: pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_test: pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    interventions: Mapping[str, Mapping[str, float | int]],
    expected_directions_dgp: Mapping[str, int],
    *,
    responsive_threshold: float = 0.005,
) -> pd.DataFrame:
    """Per-bottleneck-unit relaxation audit.

    Parameters
    ----------
    mb:
        A *fitted* :class:`MechanismBottleneckClassifier` with
        ``stage2_kind`` in ``{'sign_constrained', 'sign_constrained_augmented'}``.
        Calibrated logistic (``v0_1``) raises ``ValueError``.
    X_train, y_train:
        Same training rows used to fit ``mb`` (needed to refit stage 2).
    X_test, y_test:
        Held-out evaluation.
    interventions, expected_directions_dgp:
        Same protocol as the benchmark sweep.

    Returns
    -------
    DataFrame with columns
        ``bottleneck_unit``, ``baseline_responsive_dgp``, ``relaxed_responsive_dgp``,
        ``delta_responsive_dgp``, ``baseline_auroc``, ``relaxed_auroc``, ``delta_auroc``.
    """
    if mb.stage2_kind not in ("sign_constrained", "sign_constrained_augmented"):
        raise ValueError(
            "prior_sensitivity requires sign_constrained or sign_constrained_augmented; "
            f"got {mb.stage2_kind!r}"
        )

    y_te = np.asarray(y_test, dtype=int).ravel()

    base_proxy = _MBStage2View(mb, mb.stage2_)
    base_resp = _responsive_conformity_mb(
        base_proxy, X_test, interventions, expected_directions_dgp,
        threshold=responsive_threshold,
    )
    base_proba = mb.predict_proba(X_test)[:, 1]
    base_auc = float(roc_auc_score(y_te, base_proba))

    signs_full = mb._resolve_signs()
    rows: list[dict[str, float | str]] = []

    for j, unit in enumerate(mb.bottleneck_units_):
        relaxed = list(signs_full)
        relaxed[j] = 0
        new_stage2 = _refit_stage2_relaxed(mb, X_train, y_train, relaxed)
        proxy = _MBStage2View(mb, new_stage2)
        r_resp = _responsive_conformity_mb(
            proxy, X_test, interventions, expected_directions_dgp,
            threshold=responsive_threshold,
        )
        r_proba = proxy.predict_proba(X_test)[:, 1]
        r_auc = float(roc_auc_score(y_te, r_proba))
        rows.append({
            "bottleneck_unit": unit,
            "prior_sign": int(signs_full[j]),
            "baseline_responsive_dgp": base_resp,
            "relaxed_responsive_dgp": r_resp,
            "delta_responsive_dgp": float(r_resp - base_resp),
            "baseline_auroc": base_auc,
            "relaxed_auroc": r_auc,
            "delta_auroc": float(r_auc - base_auc),
        })

    return pd.DataFrame(rows)
