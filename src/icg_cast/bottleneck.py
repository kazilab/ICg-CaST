"""Mechanism-Bottleneck Causal Network (MB-CNet) for integrated carcinogenomics.

PLAN.md reference: section 7.5. This is the v0.1 sklearn-only skeleton; the
differentiable end-to-end version using neural ODEs / UDEs is deferred to
Milestone 7.

Architecture
------------

A two-stage Concept Bottleneck Model whose predictions of
`future_cancer_transition_event` are constrained to flow through a hidden
layer pinned to the qAOP latent state vector::

    stage 1: g_phi : omics_features  ->  hat{qAOP_state}      (multi-output regressor)
    stage 2: h_theta: hat{qAOP_state} ->  risk probability    (calibrated classifier)

Counterfactual interventions become *do-operations on bottleneck units*
instead of ad-hoc feature scaling::

    model.intervene(unit="state_auc_DNA_adducts", scale=0.5)
    risk_after = model.predict_proba(X)[:, 1]

This makes the model causally coherent by construction rather than
post-hoc, which is the manuscript's primary methodological claim.

Acceptance criteria (v0.1)
--------------------------

- `bottleneck_recovery.csv` reports per-state recovery R^2.
- Mean recovery R^2 >= 0.60 across the 10 qAOP states on the default cohort.
- Intervention-conformity score >= 0.85 across the seven `do_*` interventions
  defined in `counterfactuals.py`.
- AUROC within 0.03 of the best unconstrained multi-omics baseline.
- This module does not import torch, jax, or any heavy ML dependency.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .oracle.reference_risk_oracle import reference_risk_oracle

DEFAULT_BOTTLENECK_UNITS: tuple[str, ...] = (
    "state_final_DNA_adducts",
    "state_final_ROS",
    "state_final_inflammation",
    "state_final_epigenetic_age",
    "state_final_proliferation",
    "state_final_mutation_rate",
    "state_final_clone_fraction",
    "state_final_driver_count_proxy",
    "state_final_immune_clearance",
)


STRUCTURAL_SIGNS: dict[str, int] = {
    # signs implied by the active simulator's latent_risk structural equation
    "state_final_DNA_adducts":         +1,
    "state_final_ROS":                 +1,
    "state_final_inflammation":        +1,
    "state_final_epigenetic_age":      +1,
    "state_final_proliferation":       +1,
    "state_final_mutation_rate":       +1,   # indirect via driver_count_proxy
    "state_final_clone_fraction":      +1,
    "state_final_driver_count_proxy":  +1,
    "state_final_immune_clearance":    -1,
    # AUC variants follow the same expected signs
    "state_auc_DNA_adducts":           +1,
    "state_auc_ROS":                   +1,
    "state_auc_inflammation":          +1,
    "state_auc_proliferation":         +1,
    "state_auc_mutation_rate":         +1,
    "state_auc_immune_clearance":      -1,
}


def starter_kit_latent_risk(S: pd.DataFrame) -> np.ndarray:
    """Backward-compatible alias for the frozen v1.0 reference oracle.

    Used by intervention-augmented training to compute the structural-prior
    label under a do-intervention on bottleneck units.
    """
    return np.asarray(reference_risk_oracle(S), dtype=float)


class SignConstrainedLogisticRegression(BaseEstimator, ClassifierMixin):
    """L2-regularised logistic regression with per-coefficient sign constraints.

    The optimisation is the standard penalised negative log-likelihood, solved
    with L-BFGS-B and bound constraints: coefficients are bounded to
    ``[0, +inf)`` when the expected sign is ``+1`` and ``(-inf, 0]`` when the
    expected sign is ``-1``. The intercept is unconstrained.

    Parameters
    ----------
    coef_signs:
        Sequence of +1 / -1 / 0 with length equal to ``X.shape[1]``. ``0`` means
        unconstrained for that feature.
    C:
        Inverse regularisation strength. Defaults to 1.0.
    class_weight:
        ``None`` for unweighted, ``"balanced"`` for inverse class frequency
        weighting (matching sklearn semantics).
    sample_weight_support:
        Whether ``fit`` accepts a ``sample_weight`` argument (used by
        intervention-augmented training).
    """

    def __init__(
        self,
        coef_signs: Sequence[int],
        C: float = 1.0,
        class_weight: str | None = None,
        max_iter: int = 2000,
        tol: float = 1e-7,
    ) -> None:
        self.coef_signs = coef_signs
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.tol = tol

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> SignConstrainedLogisticRegression:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        n, p = X.shape

        signs = np.asarray(self.coef_signs, dtype=float)
        if signs.size != p:
            raise ValueError(
                f"coef_signs length ({signs.size}) must match X.shape[1] ({p})"
            )

        if self.class_weight == "balanced":
            n_pos = max(int(np.sum(y == 1)), 1)
            n_neg = max(int(np.sum(y == 0)), 1)
            base_w = np.where(y == 1, n / (2 * n_pos), n / (2 * n_neg))
        elif self.class_weight is None:
            base_w = np.ones(n)
        else:
            raise ValueError(
                f"class_weight must be None or 'balanced', got {self.class_weight!r}"
            )

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float).ravel()
            if sample_weight.size != n:
                raise ValueError("sample_weight must match number of rows")
            w = base_w * sample_weight
        else:
            w = base_w

        bounds: list[tuple[float | None, float | None]] = []
        for s in signs:
            if s > 0:
                bounds.append((0.0, None))
            elif s < 0:
                bounds.append((None, 0.0))
            else:
                bounds.append((None, None))
        bounds.append((None, None))  # intercept

        def neg_log_lik(params: np.ndarray) -> float:
            beta = params[:p]
            b = params[p]
            z = X @ beta + b
            log_loss = np.logaddexp(0.0, z) - y * z
            penalty = 0.5 * float(np.sum(beta * beta)) / max(self.C, 1e-12)
            return float(np.sum(w * log_loss)) + penalty

        def grad(params: np.ndarray) -> np.ndarray:
            beta = params[:p]
            b = params[p]
            z = X @ beta + b
            p_pred = expit(z)
            err = w * (p_pred - y)
            gb = X.T @ err + beta / max(self.C, 1e-12)
            gi = float(err.sum())
            return np.concatenate([gb, [gi]])

        x0 = np.zeros(p + 1)
        result = minimize(
            neg_log_lik,
            x0,
            jac=grad,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iter, "gtol": self.tol},
        )
        if not result.success:
            warnings.warn(
                f"SignConstrainedLogisticRegression: optimizer did not converge: {result.message}",
                RuntimeWarning,
                stacklevel=2,
            )

        self.coef_ = result.x[:p].reshape(1, -1)
        self.intercept_ = np.array([result.x[p]])
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = p
        return self

    def decision_function(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return (X @ self.coef_.ravel() + self.intercept_).ravel()

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        z = self.decision_function(X)
        p = expit(z)
        return np.column_stack([1.0 - p, p])

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def augment_with_interventions(
    S_train: pd.DataFrame,
    interventions: Mapping[str, Mapping[str, float]],
    latent_risk_fn: Callable[[pd.DataFrame], np.ndarray],
    hazard_scale: float,
    months: int,
    rng: np.random.Generator,
    samples_per_intervention: int = 1,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate synthetic augmented (S, y, weight) triples under do-interventions.

    Each augmentation transforms the bottleneck row by an intervention's
    scale factors, computes the DGP-implied latent risk via ``latent_risk_fn``,
    converts to an event probability under the simulator's cumulative-hazard
    parameterisation, and samples a binary label.

    Returns
    -------
    S_aug, y_aug, w_aug:
        Concatenated augmented bottleneck rows, sampled labels, and
        per-sample weights normalised so the augmented set carries roughly the
        same total weight as one copy of the training set per intervention.
    """
    parts_S: list[pd.DataFrame] = []
    parts_y: list[np.ndarray] = []
    parts_w: list[np.ndarray] = []

    weight_per_row = 1.0 / max(samples_per_intervention, 1)

    for spec in interventions.values():
        for _ in range(samples_per_intervention):
            S_int = S_train.copy()
            for unit, scale in spec.items():
                if unit in S_int.columns:
                    S_int[unit] = S_int[unit] * float(scale)
            risk = latent_risk_fn(S_int)
            event_prob = 1.0 - np.exp(-hazard_scale * months * risk)
            y_int = (rng.uniform(size=len(S_int)) < event_prob).astype(int)
            w_int = np.full(len(S_int), weight_per_row, dtype=float)
            parts_S.append(S_int)
            parts_y.append(y_int)
            parts_w.append(w_int)

    if not parts_S:
        return S_train.iloc[0:0].copy(), np.array([], dtype=int), np.array([], dtype=float)

    return (
        pd.concat(parts_S, axis=0, ignore_index=True),
        np.concatenate(parts_y),
        np.concatenate(parts_w),
    )


@dataclass
class _Interventions:
    """Stored do-operations applied to bottleneck units before stage 2."""

    scales: dict[str, float] = field(default_factory=dict)
    shifts: dict[str, float] = field(default_factory=dict)

    def clear(self) -> None:
        self.scales.clear()
        self.shifts.clear()

    def apply(self, bottleneck: np.ndarray, units: Sequence[str]) -> np.ndarray:
        out = bottleneck.copy()
        for i, unit in enumerate(units):
            if unit in self.scales:
                out[:, i] = out[:, i] * self.scales[unit]
            if unit in self.shifts:
                out[:, i] = out[:, i] + self.shifts[unit]
        return out


class _StandardisedSignConstrained(BaseEstimator, ClassifierMixin):
    """`SignConstrainedLogisticRegression` pre-composed with a `StandardScaler`.

    Pre-scaling brings the L2 penalty into a comparable regime across features.
    Sign constraints are invariant under positive linear scaling, so the
    standardisation does not affect feasibility.
    """

    def __init__(
        self,
        signs: Sequence[int],
        class_weight: str | None = "balanced",
        C: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        self.signs = signs
        self.class_weight = class_weight
        self.C = C
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        self.scaler_ = StandardScaler().fit(np.asarray(X, dtype=float))
        Xs = self.scaler_.transform(np.asarray(X, dtype=float))
        self.lr_ = SignConstrainedLogisticRegression(
            coef_signs=self.signs,
            C=self.C,
            class_weight=self.class_weight,
        )
        self.lr_.fit(Xs, y, sample_weight=sample_weight)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        Xs = self.scaler_.transform(np.asarray(X, dtype=float))
        return self.lr_.predict_proba(Xs)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class MechanismBottleneckClassifier(BaseEstimator, ClassifierMixin):
    """Two-stage MB-CNet.

    Parameters
    ----------
    bottleneck_units:
        Names of cohort columns that constitute the qAOP latent state vector.
        Stage 1 predicts these values from `feature_columns`. Stage 2 predicts
        risk from the predicted bottleneck only.
    feature_columns:
        Names of cohort columns used as inputs to stage 1.
    stage1_estimator:
        sklearn regressor wrapped in `MultiOutputRegressor`. Defaults to a
        small `RandomForestRegressor` for reproducibility.
    stage2_kind:
        - ``"calibrated_logistic"`` (default): the v0.1 unconstrained,
          isotonic-calibrated logistic regression.
        - ``"sign_constrained"``: ``SignConstrainedLogisticRegression`` with
          per-unit signs taken from ``coef_signs`` or, if absent, from the
          ``STRUCTURAL_SIGNS`` registry. Coefficients are constrained to the
          structurally-implied half-line.
        - ``"sign_constrained_augmented"``: like ``"sign_constrained"`` but the
          stage-2 fit is augmented with intervention-implied samples drawn
          using a user-supplied ``latent_risk_fn``.
    coef_signs:
        Optional mapping bottleneck-unit name -> ±1 / 0 used when
        ``stage2_kind`` is sign-constrained. Defaults to ``STRUCTURAL_SIGNS``.
    stage2_estimator:
        Advanced override; if provided, the wired ``stage2_kind`` is ignored.
    augment_interventions:
        For ``"sign_constrained_augmented"``: a mapping
        ``intervention_name -> {bottleneck_unit: scale}``.
    augment_latent_risk_fn:
        For ``"sign_constrained_augmented"``: a callable mapping a bottleneck
        DataFrame to a vector of latent risks; the simulator's structural
        equation is the canonical choice.
    augment_hazard_scale, augment_months, augment_samples_per_intervention:
        Cumulative-hazard parameters used by the augmenter. The defaults
        match the starter kit (`event_hazard_scale=0.020`, `months=72`).
    random_state:
        Seed forwarded to underlying estimators and the augmentation RNG.
    """

    def __init__(
        self,
        bottleneck_units: Sequence[str] = DEFAULT_BOTTLENECK_UNITS,
        feature_columns: Sequence[str] | None = None,
        stage1_estimator: BaseEstimator | None = None,
        stage2_estimator: BaseEstimator | None = None,
        stage2_kind: str = "calibrated_logistic",
        coef_signs: Mapping[str, int] | None = None,
        augment_interventions: Mapping[str, Mapping[str, float]] | None = None,
        augment_latent_risk_fn: Callable[[pd.DataFrame], np.ndarray] | None = None,
        augment_hazard_scale: float = 0.020,
        augment_months: int = 72,
        augment_samples_per_intervention: int = 1,
        random_state: int | None = 7,
    ) -> None:
        self.bottleneck_units = tuple(bottleneck_units)
        self.feature_columns = tuple(feature_columns) if feature_columns is not None else None
        self.stage1_estimator = stage1_estimator
        self.stage2_estimator = stage2_estimator
        self.stage2_kind = stage2_kind
        self.coef_signs = coef_signs
        self.augment_interventions = augment_interventions
        self.augment_latent_risk_fn = augment_latent_risk_fn
        self.augment_hazard_scale = augment_hazard_scale
        self.augment_months = augment_months
        self.augment_samples_per_intervention = augment_samples_per_intervention
        self.random_state = random_state
        self._interventions = _Interventions()

    def _stage1(self) -> BaseEstimator:
        if self.stage1_estimator is not None:
            return self.stage1_estimator
        base = RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=3,
            random_state=self.random_state,
            n_jobs=1,
        )
        # SimpleImputer transparently handles NaN-valued omics columns from
        # variants such as `partial_observability`. For NaN-free cohorts it is
        # a no-op beyond computing column means.
        return make_pipeline(
            SimpleImputer(strategy="mean", keep_empty_features=True),
            MultiOutputRegressor(base),
        )

    def _resolve_signs(self) -> list[int]:
        source = dict(self.coef_signs) if self.coef_signs else dict(STRUCTURAL_SIGNS)
        signs: list[int] = []
        missing: list[str] = []
        for u in self.bottleneck_units_:
            if u in source:
                signs.append(int(source[u]))
            else:
                missing.append(u)
                signs.append(0)
        if missing:
            warnings.warn(
                "no structural sign supplied for bottleneck units; treating as unconstrained: "
                + ", ".join(missing),
                RuntimeWarning,
                stacklevel=2,
            )
        return signs

    def _stage2(self) -> BaseEstimator:
        if self.stage2_estimator is not None:
            return self.stage2_estimator

        if self.stage2_kind == "calibrated_logistic":
            inner = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=self.random_state,
                ),
            )
            return CalibratedClassifierCV(inner, method="isotonic", cv=3)

        if self.stage2_kind in ("sign_constrained", "sign_constrained_augmented"):
            signs = self._resolve_signs()
            return _StandardisedSignConstrained(
                signs=signs, class_weight="balanced",
                random_state=self.random_state,
            )

        raise ValueError(
            f"unknown stage2_kind: {self.stage2_kind!r}. expected one of: "
            "'calibrated_logistic', 'sign_constrained', 'sign_constrained_augmented'"
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series, S: pd.DataFrame | None = None) -> MechanismBottleneckClassifier:
        """Fit MB-CNet.

        Parameters
        ----------
        X:
            Observed feature matrix (transcriptomic, epigenomic, signatures,
            host, KCC, dose). May also contain the bottleneck columns; they
            are sliced out of the stage-1 input automatically.
        y:
            Binary outcome `future_cancer_transition_event`.
        S:
            Optional ground-truth bottleneck targets. If `None`, columns
            named in `bottleneck_units` are looked up in `X` and removed from
            stage-1 inputs.
        """
        X = pd.DataFrame(X).reset_index(drop=True)
        y = np.asarray(y).astype(int).ravel()

        if S is None:
            missing = [u for u in self.bottleneck_units if u not in X.columns]
            if missing:
                raise ValueError(
                    f"bottleneck units missing from X and no S provided: {missing}"
                )
            S = X[list(self.bottleneck_units)].copy()
            feature_cols = [c for c in X.columns if c not in self.bottleneck_units]
        else:
            S = pd.DataFrame(S).reset_index(drop=True)
            feature_cols = list(X.columns)

        if self.feature_columns is not None:
            feature_cols = [c for c in feature_cols if c in self.feature_columns]
            if not feature_cols:
                raise ValueError("feature_columns did not overlap with X columns")

        self.feature_columns_ = tuple(feature_cols)
        self.bottleneck_units_ = tuple(self.bottleneck_units)

        self.stage1_ = self._stage1()
        self.stage2_ = self._stage2()

        self.stage1_.fit(X[list(self.feature_columns_)].to_numpy(), S.to_numpy())
        S_hat_arr = self.stage1_.predict(X[list(self.feature_columns_)].to_numpy())
        S_hat = pd.DataFrame(S_hat_arr, columns=list(self.bottleneck_units_))

        if self.stage2_kind == "sign_constrained_augmented":
            if self.augment_interventions is None or self.augment_latent_risk_fn is None:
                raise ValueError(
                    "stage2_kind='sign_constrained_augmented' requires "
                    "augment_interventions and augment_latent_risk_fn"
                )
            rng = np.random.default_rng(self.random_state)
            S_aug, y_aug, w_aug = augment_with_interventions(
                S,
                interventions=self.augment_interventions,
                latent_risk_fn=self.augment_latent_risk_fn,
                hazard_scale=self.augment_hazard_scale,
                months=self.augment_months,
                rng=rng,
                samples_per_intervention=self.augment_samples_per_intervention,
            )
            S_combined = pd.concat([S_hat, S_aug], axis=0, ignore_index=True)
            y_combined = np.concatenate([y, y_aug])
            w_combined = np.concatenate([np.ones(len(y), dtype=float), w_aug])
            self.stage2_.fit(S_combined, y_combined, sample_weight=w_combined)
        else:
            self.stage2_.fit(S_hat, y)

        self.classes_ = np.array([0, 1])
        return self

    def _bottleneck(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X)
        S_hat = self.stage1_.predict(X[list(self.feature_columns_)].to_numpy())
        return self._interventions.apply(S_hat, self.bottleneck_units_)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.stage2_.predict_proba(self._bottleneck(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_bottleneck(self, X: pd.DataFrame) -> pd.DataFrame:
        S_hat = self._bottleneck(X)
        return pd.DataFrame(S_hat, columns=list(self.bottleneck_units_), index=pd.DataFrame(X).index)

    def intervene(self, unit: str, scale: float | None = None, shift: float | None = None) -> None:
        """Register a do-operation on a bottleneck unit.

        Applied at predict time after stage 1. Pass `scale=None, shift=None`
        to clear interventions on that unit.
        """
        if unit not in self.bottleneck_units_:
            raise KeyError(f"unknown bottleneck unit: {unit}")
        if scale is None and shift is None:
            self._interventions.scales.pop(unit, None)
            self._interventions.shifts.pop(unit, None)
            return
        if scale is not None:
            self._interventions.scales[unit] = float(scale)
        if shift is not None:
            self._interventions.shifts[unit] = float(shift)

    def clear_interventions(self) -> None:
        self._interventions.clear()

    def score_recovery(self, X: pd.DataFrame, S_true: pd.DataFrame) -> pd.DataFrame:
        """Per-state recovery R^2 of stage 1 on a held-out cohort."""
        S_hat = pd.DataFrame(self.stage1_.predict(X[list(self.feature_columns_)].to_numpy()),
                             columns=list(self.bottleneck_units_))
        rows = []
        for unit in self.bottleneck_units_:
            r2 = r2_score(S_true[unit].to_numpy(), S_hat[unit].to_numpy())
            rows.append({"bottleneck_unit": unit, "recovery_r2": float(r2)})
        return pd.DataFrame(rows)

    def score_intervention_conformity(
        self,
        X: pd.DataFrame,
        interventions: Mapping[str, Mapping[str, float | int]],
        expected_directions: Mapping[str, int],
    ) -> tuple[float, pd.DataFrame]:
        """Fraction of interventions whose predicted-risk change has the expected sign.

        Parameters
        ----------
        X:
            Held-out features.
        interventions:
            Mapping intervention_name -> {bottleneck_unit: scale} dict.
        expected_directions:
            Mapping intervention_name -> -1 (should decrease risk) or +1 (should
            not increase risk). Use 0 to mark "no clear expectation".

        Returns
        -------
        score:
            Fraction of named interventions matching expected direction.
        table:
            Per-intervention dataframe with risk before/after, mean change,
            expected direction, and pass/fail flag.
        """
        base = self.predict_proba(X)[:, 1]
        rows: list[dict[str, object]] = []
        for name, spec in interventions.items():
            self.clear_interventions()
            for unit, scale in spec.items():
                self.intervene(unit=unit, scale=float(scale))
            after = self.predict_proba(X)[:, 1]
            delta = float(np.mean(after - base))
            expected = int(expected_directions.get(name, 0))
            if expected == 0:
                passed = True
            else:
                passed = (np.sign(delta) == np.sign(expected)) or np.isclose(delta, 0.0, atol=1e-3)
            rows.append({
                "intervention": name,
                "mean_risk_before": float(np.mean(base)),
                "mean_risk_after": float(np.mean(after)),
                "mean_risk_change": delta,
                "expected_direction": expected,
                "passed_directionality": bool(passed),
            })
        self.clear_interventions()
        table = pd.DataFrame(rows)
        scored = table[table["expected_direction"] != 0]
        score = float(scored["passed_directionality"].mean()) if len(scored) else float("nan")
        return score, table


def baseline_auroc_gap(
    bottleneck_auc: float,
    best_baseline_auc: float,
) -> float:
    """Helper: returns the absolute AUROC gap used in the v0.1 acceptance criterion."""
    return float(best_baseline_auc - bottleneck_auc)


def quick_auroc(model: MechanismBottleneckClassifier, X: pd.DataFrame, y: Iterable[int]) -> float:
    proba = model.predict_proba(X)[:, 1]
    return float(roc_auc_score(np.asarray(list(y), dtype=int), proba))
