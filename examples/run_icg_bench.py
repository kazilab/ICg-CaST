"""End-to-end ICg-Bench example: score baselines and MB-CNet on small DGPs.

This is a runnable, manuscript-shaped illustration of the benchmark. It uses a
tiny cohort size (``n = 200``, ``months = 24``) so it finishes well under a
minute on a laptop, and it never downloads or touches real data. For the
full canonical sweep, run ``icg-cast bench sweep`` instead — that command
delegates to ``scripts/bottleneck_proof_of_concept.py`` and produces the
manuscript-grade results stored under ``outputs/bottleneck_v0_5/``.

Run from the repo root:

    python examples/run_icg_bench.py

Outputs (relative to repo root) are written to ``outputs/bench_example/``:

    leaderboard.csv
    leaderboard.json
    results.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from icg_cast import DEFAULT_BOTTLENECK_UNITS, MechanismBottleneckClassifier, __version__
from icg_cast.benchmark import (
    LeaderboardEntry,
    append_entry,
    generate,
    load_variant,
    run_benchmark,
    score_summary,
    task_intervention_conformity,
    task_latent_recovery,
    task_risk_prediction,
)
from icg_cast.bottleneck import starter_kit_latent_risk

INTERVENTIONS: dict[str, dict[str, float]] = {
    "do_DNA_repair_rescue":             {"state_final_DNA_adducts":        0.55},
    "do_ROS_inflammation_blockade":     {"state_final_ROS":                0.50,
                                         "state_final_inflammation":       0.50},
    "do_epigenetic_memory_reset":       {"state_final_epigenetic_age":     0.45},
    "do_proliferation_suppression":     {"state_final_proliferation":      0.50},
    "do_immune_surveillance_restore":   {"state_final_immune_clearance":   1.50},
    "do_repair_inhibition":             {"state_final_DNA_adducts":        1.80},
    "do_artificial_proliferation":      {"state_final_proliferation":      1.80},
}

EXPECTED_DIRECTIONS: dict[str, int] = {
    "do_DNA_repair_rescue":           -1,
    "do_ROS_inflammation_blockade":   -1,
    "do_epigenetic_memory_reset":     -1,
    "do_proliferation_suppression":   -1,
    "do_immune_surveillance_restore": -1,
    "do_repair_inhibition":           +1,
    "do_artificial_proliferation":    +1,
}


def _omics_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c.startswith(("tx_", "epi_", "sig_activity_", "kcc", "host_"))
        or c in ("dose", "mut_total_count")
    ]


def _run_one(cohort_name: str, seed: int) -> list:
    variant = load_variant(cohort_name)
    df, _ = generate(cohort_name, n=200, months=24, seed=seed)

    feat = _omics_feature_columns(df)
    bcols = [c for c in DEFAULT_BOTTLENECK_UNITS if c in df.columns]
    X = df[feat].copy()
    S = df[bcols].copy()
    y = df["future_cancer_transition_event"].astype(int).to_numpy()

    if np.unique(y).size < 2:
        return []

    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)

    results = []

    # ---- baseline: plain logistic regression on omics features ---------------
    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    baseline.fit(X.iloc[tr], y[tr])
    baseline_tasks = {
        "risk_prediction": task_risk_prediction(baseline, X.iloc[te], y[te]),
    }
    baseline_result = run_benchmark(
        variant=variant,
        model_name="logistic_l2_baseline",
        package_version=__version__,
        task_outputs=baseline_tasks,
        notes="example/run_icg_bench.py, n=200, months=24",
    )
    results.append(baseline_result)

    # ---- MB-CNet sign-constrained --------------------------------------------
    mb = MechanismBottleneckClassifier(
        stage2_kind="sign_constrained_augmented",
        bottleneck_units=bcols,
        feature_columns=feat,
        augment_interventions=INTERVENTIONS,
        augment_latent_risk_fn=starter_kit_latent_risk,
        augment_hazard_scale=0.020,
        augment_months=24,
        augment_samples_per_intervention=2,
        random_state=seed,
    )
    mb.fit(X.iloc[tr].join(S.iloc[tr]), y[tr])

    mb_tasks = {
        "risk_prediction": task_risk_prediction(mb, X.iloc[te], y[te]),
        "latent_recovery": task_latent_recovery(mb, X.iloc[te], S.iloc[te]),
        "intervention_conformity": task_intervention_conformity(
            mb, X.iloc[te], INTERVENTIONS, EXPECTED_DIRECTIONS,
        ),
    }
    mb_result = run_benchmark(
        variant=variant,
        model_name="mb_cnet_sign_constrained_augmented",
        package_version=__version__,
        task_outputs=mb_tasks,
        notes="example/run_icg_bench.py, n=200, months=24",
    )
    results.append(mb_result)
    return results


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    outdir = repo_root / "outputs" / "bench_example"
    outdir.mkdir(parents=True, exist_ok=True)

    cohorts = ["linear_lowhet", "nonlinear_mixhost", "misspecified_signs"]
    all_results = []
    for cohort in cohorts:
        for seed in (7, 13):
            for result in _run_one(cohort, seed):
                entry = LeaderboardEntry.from_result(result)
                append_entry(entry, str(outdir))
                all_results.append({
                    "cohort": cohort,
                    "seed": seed,
                    "model": result.model_name,
                    "summary": score_summary(result),
                    "result": result.to_dict(),
                })

    (outdir / "results.json").write_text(
        json.dumps(all_results, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"wrote leaderboard -> {outdir / 'leaderboard.csv'}")
    print(f"wrote results -> {outdir / 'results.json'}")
    summary_rows = [
        {
            "cohort": r["cohort"],
            "seed": r["seed"],
            "model": r["model"],
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in r["summary"].items()},
        }
        for r in all_results
    ]
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
