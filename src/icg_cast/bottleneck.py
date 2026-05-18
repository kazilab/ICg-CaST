"""Mechanism-Bottleneck Causal Network (MB-CNet) for integrated carcinogenomics.

This is the v0.1 sklearn-only skeleton; a differentiable end-to-end version
using neural ODEs / UDEs is out of scope for now.

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
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .coefficients import CoefficientRegistry
from .coefficients import registry as _coefficient_registry
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


_BOTTLENECK_EFFECT_COEFFICIENTS: dict[str, str] = {
    "DNA_adducts": "dynamics.latent_risk.dna_coupling",
    "ROS": "dynamics.latent_risk.ros_coupling",
    "inflammation": "dynamics.latent_risk.inflammation_coupling",
    "epigenetic_age": "dynamics.latent_risk.epigenetic_coupling",
    "proliferation": "dynamics.latent_risk.proliferation_coupling",
    # mutation_rate is a *mediator-as-bottleneck*, not a direct latent-risk
    # term: it raises risk only through next-month driver_count_proxy in the
    # qAOP recurrence. The card's ``effect_direction`` carries the net
    # downstream sign explicitly (rather than relying on the sign of
    # ``mut_scale``, which is a positive scale constant). The sign-constrained
    # stage-2 coefficient on ``state_final_mutation_rate`` is therefore a
    # *reduced-form* directionality test, not a structural one — the same
    # ``do(mutation_rate := x)`` intervention has feedback through driver
    # accumulation in the true DGP that the stage-2 linear coefficient cannot
    # represent.
    "mutation_rate": "dynamics.mutation_rate.scale",
    "clone_fraction": "dynamics.latent_risk.clone_coupling",
    "driver_count_proxy": "dynamics.latent_risk.driver_coupling",
    "immune_clearance": "dynamics.latent_risk.immune_coupling",
}


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def structural_signs_from_registry(
    registry_obj: CoefficientRegistry | None = None,
) -> dict[str, int]:
    """Derive bottleneck sign constraints from coefficient-card metadata.

    ``effect_direction`` is the source of truth because the net biological
    direction can differ from the literal coefficient sign; for example,
    immune clearance has a positive coupling card but is subtracted in the
    latent-risk equation.
    """
    r = _coefficient_registry() if registry_obj is None else registry_obj
    signs: dict[str, int] = {}
    for state_name, coefficient_name in _BOTTLENECK_EFFECT_COEFFICIENTS.items():
        card = r.card(coefficient_name)
        direction = (
            int(card.effect_direction)
            if card.effect_direction is not None
            else _sign(float(card.default_value))
        )
        signs[f"state_final_{state_name}"] = direction
        signs[f"state_auc_{state_name}"] = direction
    return signs


def __getattr__(name: str) -> dict[str, int]:
    # Resolve ``STRUCTURAL_SIGNS`` against the active registry on every access so
    # callers under ``use_registry(...)`` or after a calibration overlay see live
    # signs rather than an import-time snapshot.
    if name == "STRUCTURAL_SIGNS":
        return structural_signs_from_registry()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def starter_kit_latent_risk(S: pd.DataFrame) -> np.ndarray:
    """Backward-compatible alias for the frozen v1.0 reference oracle.

    Used by intervention-augmented training to compute the structural-prior
    label under a do-intervention on bottleneck units.
    """
    return np.asarray(reference_risk_oracle(S), dtype=float)


def _feature_modality(column: str) -> str:
    if column.startswith("tx_"):
        return "transcriptomic"
    if column.startswith("epi_"):
        return "epigenomic"
    if column.startswith("sig"):
        return "mutational_signature"
    if column.startswith("kcc"):
        return "kcc"
    if column.startswith("host_"):
        return "host"
    if column == "dose":
        return "dose"
    return "other"


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
    latent_risk_fn: Callable[[pd.DataFrame], np.ndarray] | None,
    hazard_scale: float,
    months: int,
    rng: np.random.Generator,
    samples_per_intervention: int = 1,
    *,
    cumulative_risk_fn: Callable[[pd.DataFrame], np.ndarray] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate synthetic augmented (S, y, weight) triples under do-interventions.

    Each augmentation transforms the supplied bottleneck rows by an
    intervention's scale factors, computes the DGP-implied cumulative latent
    risk, converts to an event probability under the simulator's cumulative-
    hazard parameterisation, and samples a binary label. The caller chooses
    whether ``S_train`` is clean ground-truth state or stage-1 predicted state;
    augmented MB-CNet training passes stage-1 predictions so stage 2 sees one
    measurement regime.

    Risk parameterisation
    ---------------------
    Two mutually exclusive hooks are supported:

    - ``cumulative_risk_fn``: returns the per-subject cumulative hazard
      ``Σ_t -log(1 - p_t)`` (e.g. by re-simulating the trajectory under the
      intervention and re-applying the prob-to-hazard mapping). Event
      probability is then ``1 - exp(-hazard_scale * cumulative_risk)``,
      matching the simulator's labelling exactly.
    - ``latent_risk_fn``: returns a per-subject final-state event probability
      ``p`` (the simulator's sigmoid output). The cumulative hazard is then
      *approximated* as ``months * -log(1 - p)`` — i.e. "assume p is constant
      across the horizon" — which is dimensionally consistent with the
      simulator's redefined cumulative hazard but biased high for any
      trajectory whose per-month risk ramps up over time (the typical case).
      Use this path for sign-constrained direction tests only; stage-2
      calibration (Brier, ECE) under augmentation is unreliable.

    Exactly one of the two must be supplied.

    Returns
    -------
    S_aug, y_aug, w_aug:
        Concatenated augmented bottleneck rows, sampled labels, and
        per-sample weights normalised so all augmented interventions together
        carry roughly the same total weight as one copy of the training set.
    """
    if (cumulative_risk_fn is None) == (latent_risk_fn is None):
        raise ValueError(
            "augment_with_interventions requires exactly one of "
            "cumulative_risk_fn or latent_risk_fn"
        )

    parts_S: list[pd.DataFrame] = []
    parts_y: list[np.ndarray] = []
    parts_w: list[np.ndarray] = []

    weight_per_row = 1.0 / (
        max(samples_per_intervention, 1) * max(len(interventions), 1)
    )

    for spec in interventions.values():
        for _ in range(samples_per_intervention):
            S_int = S_train.copy()
            for unit, scale in spec.items():
                if unit in S_int.columns:
                    S_int[unit] = S_int[unit] * float(scale)
            if cumulative_risk_fn is not None:
                cumulative = np.clip(np.asarray(cumulative_risk_fn(S_int), dtype=float), 0.0, None)
            else:
                warnings.warn(
                    "augment_with_interventions: using latent_risk_fn fallback. "
                    "Cumulative hazard is approximated as months * -log(1 - p), "
                    "which is biased high for monotonically-rising trajectories. "
                    "This path is intended for sign-direction tests only; stage-2 "
                    "calibration (Brier, ECE) under augmentation will be biased. "
                    "Supply cumulative_risk_fn for calibration-grade augmentation.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                risk = np.clip(
                    np.asarray(latent_risk_fn(S_int), dtype=float),  # type: ignore[misc]
                    0.0,
                    1.0 - 1e-12,
                )
                # Constant-risk approximation: cumulative hazard = months * -log(1 - p).
                # Dimensionally consistent with the simulator's per-month-probability
                # interpretation, but biased high for monotonically-rising trajectories.
                cumulative = np.clip(-months * np.log1p(-risk), 0.0, None)
            event_prob = 1.0 - np.exp(-hazard_scale * cumulative)
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
    stage1_imputation_strategy:
        Strategy used by the default `SimpleImputer` for missing omics values.
    stage1_add_missing_indicator:
        Whether the default stage-1 pipeline appends missingness indicators so
        partial-observability cohorts can expose modality dropout to the model.
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
    strict_signs:
        Whether sign-constrained variants should fail if any bottleneck unit
        lacks a supplied or registry-derived sign. Set ``False`` to retain the
        older warning-and-unconstrained fallback for exploratory runs.
    stage2_estimator:
        Advanced override; if provided, the wired ``stage2_kind`` is ignored.
    augment_interventions:
        For ``"sign_constrained_augmented"``: a mapping
        ``intervention_name -> {bottleneck_unit: scale}``.
    augment_latent_risk_fn:
        For ``"sign_constrained_augmented"``: a callable mapping a bottleneck
        DataFrame to a vector of per-subject *final-state* latent risks. The
        augmenter then approximates cumulative risk as ``months * latent_risk``
        — a simplification because per-month state under a do-intervention is
        not available without re-simulating.
    augment_cumulative_risk_fn:
        Preferred over ``augment_latent_risk_fn`` when the caller can compute
        per-subject cumulative latent risk directly (e.g. by re-running the
        simulator under each intervention). Bypasses the ``months * risk``
        approximation and matches the simulator's labelling exactly.
    augment_hazard_scale, augment_months, augment_samples_per_intervention:
        Cumulative-hazard parameters used by the augmenter. The defaults
        match the starter kit (`event_hazard_scale=0.020`, `months=72`).
        ``augment_months`` is unused when ``augment_cumulative_risk_fn`` is
        provided.
    random_state:
        Seed forwarded to underlying estimators and the augmentation RNG.
    """

    def __init__(
        self,
        bottleneck_units: Sequence[str] = DEFAULT_BOTTLENECK_UNITS,
        feature_columns: Sequence[str] | None = None,
        stage1_estimator: BaseEstimator | None = None,
        stage1_imputation_strategy: str = "mean",
        stage1_add_missing_indicator: bool = True,
        stage2_estimator: BaseEstimator | None = None,
        stage2_kind: str = "calibrated_logistic",
        coef_signs: Mapping[str, int] | None = None,
        strict_signs: bool = True,
        augment_interventions: Mapping[str, Mapping[str, float]] | None = None,
        augment_latent_risk_fn: Callable[[pd.DataFrame], np.ndarray] | None = None,
        augment_cumulative_risk_fn: Callable[[pd.DataFrame], np.ndarray] | None = None,
        augment_hazard_scale: float = 0.020,
        augment_months: int = 72,
        augment_samples_per_intervention: int = 1,
        random_state: int | None = 7,
    ) -> None:
        self.bottleneck_units = tuple(bottleneck_units)
        self.feature_columns = tuple(feature_columns) if feature_columns is not None else None
        self.stage1_estimator = stage1_estimator
        self.stage1_imputation_strategy = stage1_imputation_strategy
        self.stage1_add_missing_indicator = stage1_add_missing_indicator
        self.stage2_estimator = stage2_estimator
        self.stage2_kind = stage2_kind
        self.coef_signs = coef_signs
        self.strict_signs = strict_signs
        self.augment_interventions = augment_interventions
        self.augment_latent_risk_fn = augment_latent_risk_fn
        self.augment_cumulative_risk_fn = augment_cumulative_risk_fn
        self.augment_hazard_scale = augment_hazard_scale
        self.augment_months = augment_months
        self.augment_samples_per_intervention = augment_samples_per_intervention
        self.random_state = random_state
        self._interventions = _Interventions()

    def _stage1(self) -> BaseEstimator:
        if self.stage1_estimator is not None:
            return clone(self.stage1_estimator)
        if self.stage1_imputation_strategy not in {"mean", "median", "most_frequent", "constant"}:
            raise ValueError(
                "stage1_imputation_strategy must be one of "
                "'mean', 'median', 'most_frequent', or 'constant'"
            )
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
            SimpleImputer(
                strategy=self.stage1_imputation_strategy,
                keep_empty_features=True,
                add_indicator=bool(self.stage1_add_missing_indicator),
            ),
            MultiOutputRegressor(base),
        )

    def _resolve_signs(self) -> list[int]:
        source = (
            dict(self.coef_signs)
            if self.coef_signs is not None
            else structural_signs_from_registry()
        )
        signs: list[int] = []
        missing: list[str] = []
        for u in self.bottleneck_units_:
            if u in source:
                signs.append(int(source[u]))
            else:
                missing.append(u)
                signs.append(0)
        if missing:
            message = (
                "no structural sign supplied for bottleneck units: "
                + ", ".join(missing)
            )
            if self.strict_signs:
                raise KeyError(message)
            warnings.warn(
                message + "; treating as unconstrained",
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
        if X.empty:
            raise ValueError("X must contain at least one row and one column")
        y_raw = np.asarray(y).ravel()
        if y_raw.size != len(X):
            raise ValueError(f"y length ({y_raw.size}) must match X rows ({len(X)})")
        try:
            y_float = y_raw.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("y must contain finite binary labels") from exc
        if not np.isfinite(y_float).all():
            raise ValueError("y must contain finite binary labels")
        if not np.all(np.isin(y_float, [0.0, 1.0])):
            raise ValueError("y must contain binary labels encoded as 0/1")
        y = y_float.astype(int)

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
            if len(S) != len(X):
                raise ValueError(f"S rows ({len(S)}) must match X rows ({len(X)})")
            missing = [u for u in self.bottleneck_units if u not in S.columns]
            if missing:
                raise ValueError(f"bottleneck units missing from S: {missing}")
            S = S[list(self.bottleneck_units)].copy()
            feature_cols = list(X.columns)

        if self.feature_columns is not None:
            feature_cols = [c for c in feature_cols if c in self.feature_columns]
            if not feature_cols:
                raise ValueError("feature_columns did not overlap with X columns")

        self.feature_columns_ = tuple(feature_cols)
        self.bottleneck_units_ = tuple(self.bottleneck_units)
        X_features = X[list(self.feature_columns_)]
        S_targets = S[list(self.bottleneck_units_)]
        non_numeric_features = [
            c for c in X_features.columns if not pd.api.types.is_numeric_dtype(X_features[c])
        ]
        if non_numeric_features:
            raise ValueError("stage-1 feature columns must be numeric: " + ", ".join(non_numeric_features))
        non_numeric_targets = [
            c for c in S_targets.columns if not pd.api.types.is_numeric_dtype(S_targets[c])
        ]
        if non_numeric_targets:
            raise ValueError("bottleneck target columns must be numeric: " + ", ".join(non_numeric_targets))
        if np.isinf(X_features.to_numpy(dtype=float)).any():
            raise ValueError("stage-1 feature columns must not contain +/-inf")
        if not np.isfinite(S_targets.to_numpy(dtype=float)).all():
            raise ValueError("bottleneck target S must contain finite numeric values")
        missing_fraction = X_features.isna().mean().astype(float)
        self.missingness_report_ = pd.DataFrame(
            {
                "feature": list(self.feature_columns_),
                "missing_fraction": missing_fraction.to_numpy(dtype=float),
                "modality": [_feature_modality(c) for c in self.feature_columns_],
            }
        )

        self.stage1_ = self._stage1()
        self.stage2_ = self._stage2()

        self.stage1_.fit(X_features.to_numpy(), S_targets.to_numpy())
        S_hat_arr = self.stage1_.predict(X_features.to_numpy())
        S_hat = pd.DataFrame(S_hat_arr, columns=list(self.bottleneck_units_))

        if self.stage2_kind == "sign_constrained_augmented":
            if self.augment_interventions is None or (
                self.augment_latent_risk_fn is None
                and self.augment_cumulative_risk_fn is None
            ):
                raise ValueError(
                    "stage2_kind='sign_constrained_augmented' requires "
                    "augment_interventions and one of augment_latent_risk_fn / "
                    "augment_cumulative_risk_fn"
                )
            rng = np.random.default_rng(self.random_state)
            S_aug, y_aug, w_aug = augment_with_interventions(
                S_hat,
                interventions=self.augment_interventions,
                latent_risk_fn=self.augment_latent_risk_fn,
                cumulative_risk_fn=self.augment_cumulative_risk_fn,
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

    def missingness_report(self) -> pd.DataFrame:
        """Return per-feature missingness observed during ``fit``."""
        if not hasattr(self, "missingness_report_"):
            raise AttributeError("missingness_report is available after fit")
        return self.missingness_report_.copy()

    def score_intervention_conformity(
        self,
        X: pd.DataFrame,
        interventions: Mapping[str, Mapping[str, float | int]],
        expected_directions: Mapping[str, int],
        responsive_threshold: float = 0.005,
    ) -> tuple[float, pd.DataFrame]:
        """Fraction of interventions whose predicted-risk change is responsive.

        Parameters
        ----------
        X:
            Held-out features.
        interventions:
            Mapping intervention_name -> {bottleneck_unit: scale} dict.
        expected_directions:
            Mapping intervention_name -> -1 (should decrease risk) or +1
            (should increase risk). Use 0 to mark "no clear expectation".
        responsive_threshold:
            Minimum absolute mean-risk change required to count as passing for
            non-zero expected directions.

        Returns
        -------
        score:
            Fraction of named interventions matching expected direction.
        table:
            Per-intervention dataframe with risk before/after, mean change,
            expected direction, and pass/fail flag.
        """
        if responsive_threshold < 0.0:
            raise ValueError("responsive_threshold must be non-negative")
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
                passed = (
                    np.sign(delta) == np.sign(expected)
                    and abs(delta) >= responsive_threshold
                )
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
