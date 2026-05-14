"""Biological coherence and pathway-attribution validation helpers.

These helpers operate on the output of
:func:`icg_cast.counterfactuals.counterfactual_tests` and on per-modality
feature-importance tables produced by ``train_baselines``. They do not fit any
models on their own — they only score and aggregate.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ..models import biological_coherence_summary

__all__ = [
    "biological_coherence_score",
    "biological_coherence_summary",
    "pathway_attribution_consistency",
]


def biological_coherence_score(counterfactual: pd.DataFrame) -> float:
    """Fraction of directional counterfactual tests whose observed sign matched."""
    summary = biological_coherence_summary(counterfactual)
    if summary.empty:
        return float("nan")
    return float(summary["biological_coherence_score"].iloc[0])


def pathway_attribution_consistency(
    importance: pd.DataFrame,
    pathway_map: Mapping[str, str],
    feature_column: str = "feature",
    importance_column: str = "permutation_importance_mean_auc_drop",
) -> pd.DataFrame:
    """Aggregate per-feature importance into per-pathway shares.

    Args:
        importance: long-form importance table (e.g. from ``train_baselines``).
        pathway_map: mapping ``feature -> pathway_name``. Features not in the
            map are pooled into an ``unmapped`` row.
        feature_column / importance_column: column names in ``importance``.

    Returns:
        DataFrame with one row per pathway, columns
        ``[pathway, n_features, total_importance, share_of_total]``. The share
        sums to 1.0 across rows when total importance is positive.
    """
    if feature_column not in importance.columns:
        raise KeyError(f"importance missing column {feature_column!r}")
    if importance_column not in importance.columns:
        raise KeyError(f"importance missing column {importance_column!r}")

    df = importance[[feature_column, importance_column]].copy()
    df["pathway"] = df[feature_column].map(pathway_map).fillna("unmapped")
    grouped = df.groupby("pathway", as_index=False).agg(
        n_features=(feature_column, "nunique"),
        total_importance=(importance_column, "sum"),
    )
    total = float(grouped["total_importance"].clip(lower=0).sum())
    if total > 0:
        grouped["share_of_total"] = grouped["total_importance"].clip(lower=0) / total
    else:
        grouped["share_of_total"] = np.nan
    return grouped.sort_values("total_importance", ascending=False).reset_index(drop=True)
