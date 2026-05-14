"""Baseline feature-set definitions and modeling helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
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
    """Return deterministic feature groups used by baseline models."""
    sets: dict[str, list[str]] = {}
    sets["chemical_KCC_host"] = [
        c for c in df.columns if c.startswith("kcc") or c.startswith("host_") or c == "dose"
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
            StandardScaler(with_mean=True),
            LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs"),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=180,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=180,
            max_features="sqrt",
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
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
            brier = brier_score_loss(y_test, np.clip(proba, 1e-6, 1 - 1e-6))
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
        "brier_score": float(brier_score_loss(y, np.clip(proba, 1e-6, 1 - 1e-6))),
        "event_rate": float(np.mean(y)),
        "mean_predicted_risk": float(np.mean(proba)),
        "n": float(len(y)),
    }


def calibration_metrics(y: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Return simple aggregate calibration diagnostics."""
    y = np.asarray(y, dtype=int)
    proba = np.asarray(proba, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_id = np.clip(np.digitize(proba, bins, right=True) - 1, 0, n_bins - 1)
    ece = 0.0
    rows = []
    for i in range(n_bins):
        mask = bin_id == i
        if not np.any(mask):
            continue
        observed = float(np.mean(y[mask]))
        predicted = float(np.mean(proba[mask]))
        weight = float(np.mean(mask))
        ece += weight * abs(observed - predicted)
        rows.append(
            {
                "bin": i,
                "bin_low": float(bins[i]),
                "bin_high": float(bins[i + 1]),
                "n": int(np.sum(mask)),
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


def biological_coherence_summary(counterfactual: pd.DataFrame) -> pd.DataFrame:
    """Summarize counterfactual directionality into one biological-coherence row."""
    scored = counterfactual[counterfactual["expected_direction"] != 0].copy()
    correct = scored["observed_direction"].to_numpy(dtype=int) == scored["expected_direction"].to_numpy(dtype=int)
    return pd.DataFrame(
        [
            {
                "tested_intervention_count": int(len(scored)),
                "correct_direction_count": int(correct.sum()),
                "biological_coherence_score": float(correct.mean()) if len(scored) else float("nan"),
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
