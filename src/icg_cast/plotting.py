"""Optional plotting helpers for demo outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_metrics(metrics: pd.DataFrame, outdir: str | Path) -> Path:
    """Write the modality AUROC bar plot and return its path."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best = metrics.sort_values("roc_auc", ascending=False).drop_duplicates("feature_set")
    best = best.sort_values("roc_auc", ascending=True)
    path = output_dir / "modality_auc.png"
    plt.figure(figsize=(9, 5))
    plt.barh(best["feature_set"], best["roc_auc"])
    plt.xlabel("ROC AUC on held-out synthetic cohort")
    plt.ylabel("Feature set")
    plt.title("ICg-CaST synthetic modality ablation")
    plt.xlim(0.45, 1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_trajectories(trajectories: dict[str, pd.DataFrame], outdir: str | Path) -> Path | None:
    """Write a representative latent-risk trajectory plot, if trajectories exist."""
    if not trajectories:
        return None

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for archetype, traj in trajectories.items():
        tmp = traj[["month", "latent_risk"]].copy()
        tmp["chemical_archetype"] = archetype
        rows.append(tmp)
    data = pd.concat(rows, ignore_index=True)

    path = output_dir / "example_state_trajectories.png"
    plt.figure(figsize=(9, 5))
    for archetype, sub in data.groupby("chemical_archetype"):
        plt.plot(sub["month"], sub["latent_risk"], label=archetype)
    plt.xlabel("Simulated month")
    plt.ylabel("Latent transition risk")
    plt.title("Representative qAOP/clonal-risk trajectories")
    plt.legend(fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path
