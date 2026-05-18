"""Baseline feature-set definitions and modeling helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET_DERIVED_COLUMNS: tuple[str, ...] = (
    "future_event_probability",
    "high_risk_transition_state",
    "state_final_latent_risk",
    "state_auc_latent_risk",
)


def validate_no_target_leakage(
    features: Sequence[str],
    target: str = "future_cancer_transition_event",
    feature_set: str = "feature set",
) -> None:
    """Raise when a model feature list contains endpoint-derived columns."""
    forbidden = {target, *TARGET_DERIVED_COLUMNS}
    leaky = sorted(
        {
            col
            for col in features
            if col in forbidden or col.startswith("future_") or col == "high_risk_transition_state" or "latent_risk" in col
        }
    )
    if leaky:
        raise ValueError(f"{feature_set} contains target-derived feature(s): {', '.join(leaky)}")


def feature_sets(df: pd.DataFrame, target: str = "future_cancer_transition_event") -> dict[str, list[str]]:
    """Return deterministic feature groups used by baseline models.

    ``qAOP_state`` intentionally preserves the historical combined final-state
    plus AUC state view for continuity, even though those two summaries can be
    strongly collinear. ``qAOP_state_final`` and ``qAOP_state_auc`` expose the
    split views for reports that need to inspect that sensitivity directly.
    """
    sets: dict[str, list[str]] = {}
    sets["chemical_KCC_host"] = [
        c for c in df.columns if c.startswith("kcc") or c.startswith("host_") or c == "dose"
    ]
    sets["qAOP_state_final"] = [
        c
        for c in df.columns
        if c.startswith("state_final_") and "latent_risk" not in c
    ]
    sets["qAOP_state_auc"] = [
        c
        for c in df.columns
        if c.startswith("state_auc_") and "latent_risk" not in c
    ]
    sets["qAOP_state"] = [
        c
        for c in df.columns
        if (c.startswith("state_final_") or c.startswith("state_auc_")) and "latent_risk" not in c
    ]
    sets["transcriptomic"] = [c for c in df.columns if c.startswith("tx_")]
    sets["epigenomic"] = [c for c in df.columns if c.startswith("epi_")]
    sets["mutational_signature"] = [c for c in df.columns if c.startswith("sig_activity_") or c == "mut_total_count"]
    sets["multiomics_plus_qAOP"] = sorted(set().union(*sets.values()))
    for set_name, cols in sets.items():
        validate_no_target_leakage(cols, target=target, feature_set=set_name)
    return sets


def validate_binary_target(y: np.ndarray, context: str = "target") -> None:
    classes = np.unique(y)
    if classes.size < 2:
        raise ValueError(f"{context} has one class ({classes[0]!r}); increase n/months or change seed")


def _baseline_models(seed: int) -> dict[str, object]:
    return {
        "logistic_l2": make_pipeline(
            SimpleImputer(strategy="mean", add_indicator=True),
            StandardScaler(with_mean=True),
            LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs"),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="mean", add_indicator=True),
            RandomForestClassifier(
                n_estimators=180,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            ),
        ),
        "extra_trees": make_pipeline(
            SimpleImputer(strategy="mean", add_indicator=True),
            ExtraTreesClassifier(
                n_estimators=180,
                max_features="sqrt",
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            ),
        ),
    }


def train_baselines(
    df: pd.DataFrame,
    seed: int,
    target: str = "future_cancer_transition_event",
    test_size: float = 0.30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Train baseline models and return tables plus a serializable model bundle."""
    from . import __version__
    from .counterfactuals import counterfactual_tests

    y = df[target].astype(int).to_numpy()
    validate_binary_target(y)
    fsets = feature_sets(df, target=target)
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)

    metrics = []
    best_model = None
    best_auc = -np.inf
    best_cols: list[str] = []
    best_name = ""

    for set_name, cols in fsets.items():
        x_train = df.loc[train_idx, cols]
        x_test = df.loc[test_idx, cols]
        y_train = y[train_idx]
        y_test = y[test_idx]
        validate_binary_target(y_train, context=f"{set_name} train target")
        validate_binary_target(y_test, context=f"{set_name} test target")

        for model_name, model in _baseline_models(seed).items():
            model.fit(x_train, y_train)
            proba = model.predict_proba(x_test)[:, 1]
            auc = roc_auc_score(y_test, proba)
            ap = average_precision_score(y_test, proba)
            brier = brier_score_loss(y_test, proba)
            metrics.append(
                {
                    "feature_set": set_name,
                    "model": model_name,
                    "n_features": len(cols),
                    "roc_auc": auc,
                    "average_precision": ap,
                    "brier_score": brier,
                }
            )
            if set_name == "multiomics_plus_qAOP" and auc > best_auc:
                best_auc = auc
                best_model = model
                best_cols = cols
                best_name = model_name

    if best_model is None:
        raise RuntimeError("no multiomics_plus_qAOP model was fit")
    test_df = df.iloc[test_idx].copy()
    x_test = test_df[best_cols]
    y_test = test_df[target].astype(int).to_numpy()
    perm = permutation_importance(best_model, x_test, y_test, n_repeats=4, random_state=seed, scoring="roc_auc")
    importance = pd.DataFrame(
        {
            "feature": best_cols,
            "permutation_importance_mean_auc_drop": perm.importances_mean,
            "permutation_importance_sd": perm.importances_std,
            "best_multiomics_model": best_name,
        }
    ).sort_values("permutation_importance_mean_auc_drop", ascending=False)

    counterfactual = counterfactual_tests(best_model, test_df, best_cols)
    metrics_df = pd.DataFrame(metrics).sort_values(["roc_auc", "average_precision"], ascending=False)
    bundle: dict[str, object] = {
        "model": best_model,
        "model_name": best_name,
        "feature_set": "multiomics_plus_qAOP",
        "feature_columns": best_cols,
        "target": target,
        "seed": seed,
        "test_size": test_size,
        "train_index": train_idx.tolist(),
        "test_index": test_idx.tolist(),
        "package_version": __version__,
    }
    return metrics_df, importance, counterfactual, bundle


def train_and_evaluate(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Backward-compatible helper returning metrics, importance, and counterfactual table."""
    metrics, importance, counterfactual, _bundle = train_baselines(df, seed=seed)
    return metrics, importance, counterfactual


def predictive_metrics(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Binary predictive metrics for a fitted classifier."""
    y = np.asarray(y, dtype=int)
    proba = np.asarray(proba, dtype=float)
    validate_binary_target(y, context="evaluation target")
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "average_precision": float(average_precision_score(y, proba)),
        "brier_score": float(brier_score_loss(y, proba)),
        "event_rate": float(np.mean(y)),
        "mean_predicted_risk": float(np.mean(proba)),
        "n": float(len(y)),
    }


def calibration_metrics(y: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Return simple aggregate calibration diagnostics.

    The table always contains ``n_bins`` bin rows plus one summary row. Empty
    bins retain their skeleton with ``n=0`` and NaN rates, matching
    ``validation.calibration.calibration_curve``. Bins use the conventional
    left-closed, right-open intervals ``[low, high)``, except the final bin
    includes ``proba == 1``.
    """
    y = np.asarray(y, dtype=int)
    proba = np.asarray(proba, dtype=float)
    if not np.isfinite(proba).all() or np.any((proba < 0.0) | (proba > 1.0)):
        raise ValueError("proba must lie in [0, 1]")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_id = np.digitize(proba, bins[1:-1], right=False)
    ece = 0.0
    rows = []
    for i in range(n_bins):
        mask = bin_id == i
        count = int(np.sum(mask))
        if count:
            observed = float(np.mean(y[mask]))
            predicted = float(np.mean(proba[mask]))
            weight = float(np.mean(mask))
            ece += weight * abs(observed - predicted)
        else:
            observed = np.nan
            predicted = np.nan
        rows.append(
            {
                "bin": i,
                "bin_low": float(bins[i]),
                "bin_high": float(bins[i + 1]),
                "n": count,
                "observed_event_rate": observed,
                "mean_predicted_risk": predicted,
            }
        )
    summary = {
        "bin": "summary",
        "bin_low": np.nan,
        "bin_high": np.nan,
        "n": int(len(y)),
        "observed_event_rate": float(np.mean(y)),
        "mean_predicted_risk": float(np.mean(proba)),
        "expected_calibration_error": float(ece),
    }
    return pd.concat([pd.DataFrame(rows), pd.DataFrame([summary])], ignore_index=True)


def _weighted_directional_coherence(
    correct: np.ndarray,
    weights: np.ndarray,
    *,
    context: str,
) -> float:
    weights = np.asarray(weights, dtype=float)
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError(f"{context} weights must be finite and non-negative")
    denom = float(weights.sum())
    return float(weights[correct].sum() / denom) if denom > 0 else float("nan")


def biological_coherence_summary(
    counterfactual: pd.DataFrame,
    severity_column: str = "intervention_severity_weight",
) -> pd.DataFrame:
    """Summarize counterfactual directionality into one biological-coherence row."""
    scored = counterfactual[counterfactual["expected_direction"] != 0].copy()
    if scored.empty:
        return pd.DataFrame([{ "tested_intervention_count": 0,
                               "correct_direction_count": 0,
                               "biological_coherence_score": float("nan"),
                               "effect_weighted_coherence": float("nan"),
                               "severity_weighted_coherence": float("nan"),
                               "severity_effect_weighted_coherence": float("nan") }])
    correct = scored["observed_direction"].to_numpy(dtype=int) == scored["expected_direction"].to_numpy(dtype=int)
    deltas: np.ndarray | None = None
    if "mean_absolute_risk_change" in scored.columns:
        deltas = scored["mean_absolute_risk_change"].abs().to_numpy(dtype=float)
        effect_weighted = _weighted_directional_coherence(
            correct,
            deltas,
            context="mean_absolute_risk_change",
        )
    else:
        effect_weighted = float("nan")
    if severity_column in scored.columns:
        severity_weights = scored[severity_column].to_numpy(dtype=float)
        severity_weighted = _weighted_directional_coherence(
            correct,
            severity_weights,
            context=severity_column,
        )
        severity_effect_weighted = (
            _weighted_directional_coherence(
                correct,
                severity_weights * deltas,
                context=f"{severity_column} * mean_absolute_risk_change",
            )
            if deltas is not None
            else float("nan")
        )
    else:
        severity_weighted = float("nan")
        severity_effect_weighted = float("nan")
    return pd.DataFrame(
        [
            {
                "tested_intervention_count": int(len(scored)),
                "correct_direction_count": int(correct.sum()),
                "biological_coherence_score": float(correct.mean()) if len(scored) else float("nan"),
                "effect_weighted_coherence": effect_weighted,
                "severity_weighted_coherence": severity_weighted,
                "severity_effect_weighted_coherence": severity_effect_weighted,
            }
        ]
    )


def evaluate_bundle(
    df: pd.DataFrame,
    bundle: dict[str, object],
    use_heldout: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate a saved baseline model bundle on a cohort."""
    from .counterfactuals import counterfactual_tests

    feature_columns = list(bundle["feature_columns"])
    target = str(bundle.get("target", "future_cancer_transition_event"))
    validate_no_target_leakage(feature_columns, target=target, feature_set="model bundle")
    if use_heldout and bundle.get("test_index") is not None:
        eval_df = df.iloc[list(bundle["test_index"])].copy()
    else:
        eval_df = df.copy()

    model = bundle["model"]
    y = eval_df[target].astype(int).to_numpy()
    proba = model.predict_proba(eval_df[feature_columns])[:, 1]
    metrics = pd.DataFrame([predictive_metrics(y, proba)])
    calibration = calibration_metrics(y, proba)
    counterfactual = counterfactual_tests(model, eval_df, feature_columns)
    coherence = biological_coherence_summary(counterfactual)
    return metrics, calibration, counterfactual, coherence
