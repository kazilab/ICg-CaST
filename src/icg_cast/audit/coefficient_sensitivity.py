"""Per-coefficient sensitivity audit for Milestone 12."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from icg_cast.coefficients import CoefficientCard, CoefficientRegistry, use_registry
from icg_cast.coefficients import registry as default_registry

MetricFn = Callable[[pd.DataFrame], dict[str, float]]
SimulateFn = Callable[[Any], tuple[pd.DataFrame, dict[str, pd.DataFrame]]]


def default_sensitivity_metrics(cohort: pd.DataFrame) -> dict[str, float]:
    """Small downstream metric set used by the coefficient sensitivity CLI."""
    return {
        "event_probability_mean": float(cohort["future_event_probability"].mean()),
        "latent_risk_mean": float(cohort["state_final_latent_risk"].mean()),
        "dna_adducts_mean": float(cohort["state_final_DNA_adducts"].mean()),
        "ros_mean": float(cohort["state_final_ROS"].mean()),
        "clone_fraction_mean": float(cohort["state_final_clone_fraction"].mean()),
    }


def _scale_value(value: object, scale: float) -> object:
    if isinstance(value, str):
        return value
    if isinstance(value, tuple):
        arr = np.asarray(value, dtype=float) * scale
        if np.all((0.0 <= np.asarray(value, dtype=float)) & (np.asarray(value, dtype=float) <= 1.0)):
            arr = np.clip(arr, 0.0, 1.0)
        return tuple(float(x) for x in arr)
    scaled = float(value) * scale
    if isinstance(value, int):
        return int(round(max(0.0, scaled)))
    if 0.0 <= float(value) <= 1.0:
        return float(np.clip(scaled, 0.0, 1.0))
    return scaled


def _scaled_registry(
    base: CoefficientRegistry,
    card: CoefficientCard,
    scale: float,
) -> CoefficientRegistry:
    return base.replace_cards([replace(card, default_value=_scale_value(card.default_value, scale))])


def _config_with_n(base_config: Any, n_samples: int | None) -> Any:
    if n_samples is None or not hasattr(base_config, "n"):
        return base_config
    return replace(base_config, n=int(n_samples))


def run_coefficient_sensitivity_audit(
    simulate_fn: SimulateFn,
    base_config: Any,
    metrics_fn: MetricFn = default_sensitivity_metrics,
    coefficients_to_test: Sequence[str] | None = None,
    output_dir: str | Path = "outputs/audit",
    n_samples: int | None = 300,
    scales: Sequence[float] = (0.5, 1.0, 2.0),
) -> pd.DataFrame:
    """Sweep each coefficient and record downstream metric movement.

    For each numeric coefficient card, this evaluates ``scales`` against the
    same simulation config and seed. Relative change is measured against that
    coefficient's own 1x run, so the audit is deterministic for a fixed config.
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    base_registry = default_registry()
    names = list(coefficients_to_test) if coefficients_to_test is not None else base_registry.names()
    config = _config_with_n(base_config, n_samples)

    rows: list[dict[str, object]] = []
    for name in names:
        if name not in base_registry:
            continue
        card = base_registry.card(name)
        if card.is_string:
            continue

        metric_by_scale: dict[float, dict[str, float]] = {}
        for scale in scales:
            active = _scaled_registry(base_registry, card, float(scale))
            with use_registry(active):
                cohort, _ = simulate_fn(config)
            metric_by_scale[float(scale)] = metrics_fn(cohort)

        baseline = metric_by_scale.get(1.0)
        if baseline is None:
            baseline_scale = min(metric_by_scale, key=lambda s: abs(s - 1.0))
            baseline = metric_by_scale[baseline_scale]

        for scale, metrics in metric_by_scale.items():
            for metric_name, value in metrics.items():
                base_value = float(baseline[metric_name])
                denom = max(abs(base_value), 1e-8)
                rows.append(
                    {
                        "coefficient": name,
                        "scale": scale,
                        "metric": metric_name,
                        "value": float(value),
                        "baseline_value": base_value,
                        "relative_change": (float(value) - base_value) / denom,
                        "evidence_level": card.evidence_level,
                        "source": card.source,
                    }
                )

    df = pd.DataFrame(rows)
    sensitivity_path = outdir / "coefficient_sensitivity.csv"
    df.to_csv(sensitivity_path, index=False)
    _generate_sensitivity_heatmap(df, outdir / "coefficient_sensitivity_heatmap.png")
    _auto_flag_coefficients(df, outdir / "coefficient_flags.csv")
    return df


def _generate_sensitivity_heatmap(df: pd.DataFrame, path: Path) -> None:
    """Generate a compact heatmap of max absolute relative changes."""
    if df.empty:
        path.write_text("No sensitivity rows generated.\n", encoding="utf-8")
        return
    pivot = (
        df.assign(abs_change=df["relative_change"].abs())
        .pivot_table(index="coefficient", columns="metric", values="abs_change", aggfunc="max")
        .fillna(0.0)
    )
    top = pivot.max(axis=1).sort_values(ascending=False).head(40).index
    pivot = pivot.loc[top]
    fig_height = max(4.0, min(16.0, 0.28 * len(pivot) + 1.5))
    fig, ax = plt.subplots(figsize=(10.0, fig_height))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(len(pivot.columns)), labels=list(pivot.columns), rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), labels=list(pivot.index))
    ax.set_title("Coefficient sensitivity: max |relative change|")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _auto_flag_coefficients(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    flags: list[dict[str, object]] = []
    if not df.empty:
        summaries: list[dict[str, object]] = []
        for coeff, sub in df.groupby("coefficient"):
            max_change = float(sub["relative_change"].abs().max())
            evidence_level = str(sub["evidence_level"].iloc[0])
            source = str(sub["source"].iloc[0])
            summaries.append(
                {
                    "coefficient": coeff,
                    "max_change": max_change,
                    "evidence_level": evidence_level,
                    "source": source,
                }
            )
            if max_change < 0.01:
                flags.append(
                    {
                        "coefficient": coeff,
                        "flag": "CANDIDATE_FOR_REMOVAL",
                        "max_change": max_change,
                        "evidence_level": evidence_level,
                        "source": source,
                        "requires_action": False,
                        "reason": "<1% movement on every audited metric",
                    }
                )

        top_load_bearing = (
            pd.DataFrame(summaries)
            .query("max_change > 0.20")
            .sort_values("max_change", ascending=False)
            .head(10)
        )
        for row in top_load_bearing.to_dict(orient="records"):
            evidence_level = str(row["evidence_level"])
            flags.append(
                {
                    "coefficient": row["coefficient"],
                    "flag": "LOAD_BEARING",
                    "max_change": float(row["max_change"]),
                    "evidence_level": evidence_level,
                    "source": row["source"],
                    "requires_action": bool(evidence_level not in {"E1", "E2"}),
                    "reason": "top-10 coefficient with >20% movement on at least one audited metric",
                }
            )
    flags_df = pd.DataFrame(
        flags,
        columns=[
            "coefficient",
            "flag",
            "max_change",
            "evidence_level",
            "source",
            "requires_action",
            "reason",
        ],
    )
    flags_df.to_csv(path, index=False)
    return flags_df


def identify_load_bearing_coefficients(
    sensitivity_df: pd.DataFrame,
    threshold: float = 0.20,
) -> list[str]:
    """Return coefficients with more than ``threshold`` absolute movement."""
    if sensitivity_df.empty:
        return []
    max_change = sensitivity_df.groupby("coefficient")["relative_change"].apply(lambda s: s.abs().max())
    return max_change[max_change > threshold].index.tolist()
