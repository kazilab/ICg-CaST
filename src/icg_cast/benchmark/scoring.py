"""Aggregate ICg-Bench tasks into a single benchmark result.

The four scored tasks have different units, so v0.1 reports them separately
rather than collapsing into a single scalar. A composite score is provided
only as a convenience and is explicitly *not* the primary leaderboard metric.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .dgp import DGPVariant


@dataclass
class BenchmarkResult:
    """All scored task outputs for a single (variant, model) pair."""

    variant_name: str
    variant_hash: str
    model_name: str
    package_version: str
    risk_prediction: dict[str, float] = field(default_factory=dict)
    latent_recovery: dict[str, float] = field(default_factory=dict)
    intervention_conformity: dict[str, float] = field(default_factory=dict)
    cross_host_generalization: dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def to_row(self) -> dict[str, object]:
        out: dict[str, object] = {
            "variant_name": self.variant_name,
            "variant_hash": self.variant_hash,
            "model_name": self.model_name,
            "package_version": self.package_version,
        }
        for prefix, payload in (
            ("rp", self.risk_prediction),
            ("lr", self.latent_recovery),
            ("ic", self.intervention_conformity),
            ("xh", self.cross_host_generalization),
        ):
            for k, v in payload.items():
                out[f"{prefix}__{k}"] = v
        if self.notes:
            out["notes"] = self.notes
        return out

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def score_summary(result: BenchmarkResult) -> dict[str, float]:
    """Compact, human-readable per-task summary.

    Each task is summarised by one signed scalar:

    - risk_prediction -> auroc
    - latent_recovery -> r2_mean
    - intervention_conformity -> conformity_score
    - cross_host_generalization -> auroc_target (and transfer_gap)

    A composite arithmetic mean of (auroc, r2_mean, conformity_score) is also
    reported under `composite` for ranking convenience. It is NOT the
    canonical metric and should never be reported in isolation.
    """
    auroc = float(result.risk_prediction.get("auroc", np.nan))
    r2_mean = float(result.latent_recovery.get("r2_mean", np.nan))
    conformity = float(result.intervention_conformity.get("conformity_score", np.nan))
    target_auroc = float(result.cross_host_generalization.get("auroc_target", np.nan))
    gap = float(result.cross_host_generalization.get("transfer_gap", np.nan))

    parts = [v for v in (auroc, r2_mean, conformity) if np.isfinite(v)]
    composite = float(np.mean(parts)) if parts else float("nan")

    return {
        "auroc": auroc,
        "r2_mean": r2_mean,
        "conformity": conformity,
        "auroc_target": target_auroc,
        "transfer_gap": gap,
        "composite": composite,
    }


def run_benchmark(
    variant: DGPVariant,
    model_name: str,
    package_version: str,
    task_outputs: Mapping[str, Mapping[str, float]],
    notes: str = "",
) -> BenchmarkResult:
    """Assemble a `BenchmarkResult` from already-computed task dicts.

    `task_outputs` should be a mapping like::

        {
            "risk_prediction": {...},
            "latent_recovery": {...},
            "intervention_conformity": {...},
            "cross_host_generalization": {...},
        }

    Missing entries are stored as empty dicts. The caller is responsible for
    deciding which tasks to score for a given (variant, model) pair; some
    models (e.g. plain logistic regression on raw features) cannot be scored
    on `latent_recovery` or `intervention_conformity` and should leave those
    empty.
    """
    return BenchmarkResult(
        variant_name=variant.name,
        variant_hash=variant.hash(),
        model_name=model_name,
        package_version=package_version,
        risk_prediction=dict(task_outputs.get("risk_prediction", {})),
        latent_recovery=dict(task_outputs.get("latent_recovery", {})),
        intervention_conformity=dict(task_outputs.get("intervention_conformity", {})),
        cross_host_generalization=dict(task_outputs.get("cross_host_generalization", {})),
        notes=notes,
    )


def results_to_dataframe(results: list[BenchmarkResult]) -> pd.DataFrame:
    """Convert a list of results into a flat CSV-ready dataframe."""
    return pd.DataFrame([r.to_row() for r in results])
