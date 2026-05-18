"""MB-CNet proof-of-concept across variants, cohorts, and seeds.

Trains and evaluates three MechanismBottleneckClassifier variants on four
ICg-Bench DGP cohorts, replicated across seeds {7, 13, 31}. Writes
per-experiment artifacts plus aggregated mean ± SD summaries.

Variants
--------

- ``v0_1``                       : v0.1 unconstrained calibrated logistic stage 2.
- ``sign_constrained``           : sign-constrained logistic stage 2 (signs from STRUCTURAL_SIGNS).
- ``sign_constrained_augmented`` : sign-constrained + intervention-augmented training using the
                                   starter-kit structural equation.

Cohorts
-------

- ``linear_lowhet``         : discrete archetypes, low host heterogeneity, linear coupling, full omics.
- ``nonlinear_mixhost``     : Dirichlet KCC mixtures, non-linear coupling, high host heterogeneity, full omics.
- ``partial_observability`` : nonlinear_mixhost + per-subject random masking of tx_*/epi_* at 30%.
- ``nonlinear_obs``         : linear coupling but non-linear observation operator. Stresses recovery R^2.

Outputs are written under ``outputs/bottleneck_v0_5/``.

- Six cohorts (incl. ``misspecified_signs_v2``) × three MB-CNet variants × three seeds.
- Per-experiment **bootstrap 95% CIs** (row resampling of the held-out set) for the three conformity scores.

Run::

    python3 scripts/bottleneck_proof_of_concept.py
    python3 scripts/bottleneck_proof_of_concept.py --bootstrap 250   # tighter CIs (slower)
    ICG_N_BOOTSTRAP=80 python3 scripts/bottleneck_proof_of_concept.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from icg_cast.benchmark import generate  # noqa: E402
from icg_cast.benchmark.conformity_bootstrap import bootstrap_conformity  # noqa: E402
from icg_cast.bottleneck import (  # noqa: E402
    DEFAULT_BOTTLENECK_UNITS,
    MechanismBottleneckClassifier,
    starter_kit_latent_risk,
)
from icg_cast.io import ensure_dir  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


SEEDS: tuple[int, ...] = (7, 13, 31)
COHORTS: tuple[str, ...] = (
    "linear_lowhet",
    "nonlinear_mixhost",
    "partial_observability",
    "nonlinear_obs",
    "misspecified_signs",
    "misspecified_signs_v2",
)
VARIANTS: tuple[str, ...] = ("v0_1", "sign_constrained", "sign_constrained_augmented")
DEFAULT_OUTDIR = REPO_ROOT / "outputs" / "bottleneck_v0_5"

# Default bootstrap draws for 95% equal-tailed CIs on conformity scores.
# Override with ``--bootstrap N`` or env ``ICG_N_BOOTSTRAP`` (higher = slower sweep).
_DEFAULT_N_BOOTSTRAP: int = 120

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

# Prior-implied expected directions (matches STRUCTURAL_SIGNS in bottleneck.py).
EXPECTED_DIRECTIONS_PRIOR: dict[str, int] = {
    "do_DNA_repair_rescue":           -1,
    "do_ROS_inflammation_blockade":   -1,
    "do_epigenetic_memory_reset":     -1,
    "do_proliferation_suppression":   -1,
    "do_immune_surveillance_restore": -1,
    "do_repair_inhibition":           +1,
    "do_artificial_proliferation":    +1,
}

# Per-variant DGP-implied expected directions. Used to test whether each MB-CNet
# variant matches the *truth* of its source cohort, not just the structural prior.
# All variants except `misspecified_signs` share the prior directions.
EXPECTED_DIRECTIONS_DGP: dict[str, dict[str, int]] = {
    "linear_lowhet":         dict(EXPECTED_DIRECTIONS_PRIOR),
    "nonlinear_mixhost":     dict(EXPECTED_DIRECTIONS_PRIOR),
    "partial_observability": dict(EXPECTED_DIRECTIONS_PRIOR),
    "nonlinear_obs":         dict(EXPECTED_DIRECTIONS_PRIOR),
    "misspecified_signs": {
        **EXPECTED_DIRECTIONS_PRIOR,
        "do_epigenetic_memory_reset":     +1,
    },
    "misspecified_signs_v2": {
        **EXPECTED_DIRECTIONS_PRIOR,
        "do_epigenetic_memory_reset":       +1,
        "do_immune_surveillance_restore":   +1,
    },
}


RESPONSIVE_THRESHOLD: float = 0.005


@dataclass
class ExperimentResult:
    cohort: str
    variant: str
    seed: int
    auroc: float
    auprc: float
    brier: float
    r2_mean: float
    r2_per_state: dict[str, float]
    prior_conformity: float
    dgp_conformity: float
    responsive_dgp_conformity: float
    prior_conformity_ci_lo: float
    prior_conformity_ci_hi: float
    dgp_conformity_ci_lo: float
    dgp_conformity_ci_hi: float
    responsive_dgp_ci_lo: float
    responsive_dgp_ci_hi: float
    per_intervention: pd.DataFrame
    best_baseline_auroc: float
    generate_seconds: float
    baseline_seconds: float
    fit_seconds: float
    predict_seconds: float
    bootstrap_seconds: float
    total_seconds: float


def _omics_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c.startswith(("tx_", "epi_", "sig_activity_", "kcc", "host_"))
        or c == "dose"
        or c == "mut_total_count"
    ]


def _baseline_models(seed: int) -> dict[str, object]:
    return {
        "logistic_l2": make_pipeline(
            SimpleImputer(strategy="mean", keep_empty_features=True),
            StandardScaler(with_mean=True),
            LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="mean", keep_empty_features=True),
            RandomForestClassifier(
                n_estimators=200, min_samples_leaf=3,
                class_weight="balanced", random_state=seed, n_jobs=1,
            ),
        ),
        "extra_trees": make_pipeline(
            SimpleImputer(strategy="mean", keep_empty_features=True),
            ExtraTreesClassifier(
                n_estimators=200, min_samples_leaf=3, max_features="sqrt",
                class_weight="balanced", random_state=seed, n_jobs=1,
            ),
        ),
    }


def _baselines(X_tr, y_tr, X_te, y_te, seed: int) -> pd.DataFrame:
    rows = []
    for name, m in _baseline_models(seed).items():
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_te)[:, 1]
        rows.append({
            "model": name,
            "auroc": float(roc_auc_score(y_te, p)),
            "auprc": float(average_precision_score(y_te, p)),
            "brier": float(brier_score_loss(y_te, np.clip(p, 1e-6, 1 - 1e-6))),
        })
    return pd.DataFrame(rows).sort_values("auroc", ascending=False).reset_index(drop=True)


def _make_variant(variant: str, feature_cols: list[str], bottleneck_cols: list[str], seed: int) -> MechanismBottleneckClassifier:
    kwargs = dict(
        bottleneck_units=bottleneck_cols,
        feature_columns=feature_cols,
        random_state=seed,
    )
    if variant == "v0_1":
        return MechanismBottleneckClassifier(stage2_kind="calibrated_logistic", **kwargs)
    if variant == "sign_constrained":
        return MechanismBottleneckClassifier(stage2_kind="sign_constrained", **kwargs)
    if variant == "sign_constrained_augmented":
        return MechanismBottleneckClassifier(
            stage2_kind="sign_constrained_augmented",
            augment_interventions=INTERVENTIONS,
            augment_latent_risk_fn=starter_kit_latent_risk,
            augment_hazard_scale=0.020,
            augment_months=72,
            augment_samples_per_intervention=2,
            **kwargs,
        )
    raise ValueError(variant)


def _run_one(
    cohort_name: str,
    variant: str,
    seed: int,
    *,
    n_bootstrap: int,
) -> ExperimentResult:
    total_t0 = perf_counter()
    generate_t0 = perf_counter()
    df, _ = generate(cohort_name, n=1200, months=72, seed=seed)
    generate_seconds = perf_counter() - generate_t0
    feature_cols = _omics_feature_columns(df)
    bottleneck_cols = [c for c in DEFAULT_BOTTLENECK_UNITS if c in df.columns]

    X = df[feature_cols].copy()
    S = df[bottleneck_cols].copy()
    y = df["future_cancer_transition_event"].astype(int).to_numpy()

    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)

    baseline_t0 = perf_counter()
    bl = _baselines(X.iloc[tr], y[tr], X.iloc[te], y[te], seed=seed)
    baseline_seconds = perf_counter() - baseline_t0
    best_bl_auc = float(bl["auroc"].max())

    mb = _make_variant(variant, feature_cols, bottleneck_cols, seed=seed)
    fit_t0 = perf_counter()
    mb.fit(X.iloc[tr].join(S.iloc[tr]), y[tr])
    fit_seconds = perf_counter() - fit_t0

    predict_t0 = perf_counter()
    proba = mb.predict_proba(X.iloc[te])[:, 1]
    predict_seconds = perf_counter() - predict_t0
    auc = float(roc_auc_score(y[te], proba))
    ap = float(average_precision_score(y[te], proba))
    br = float(brier_score_loss(y[te], np.clip(proba, 1e-6, 1 - 1e-6)))

    rec = mb.score_recovery(X.iloc[te], S.iloc[te])
    r2_mean = float(rec["recovery_r2"].mean())
    r2_dict = dict(zip(rec["bottleneck_unit"], rec["recovery_r2"].astype(float), strict=False))

    prior_conf, prior_table = mb.score_intervention_conformity(
        X.iloc[te], interventions=INTERVENTIONS,
        expected_directions=EXPECTED_DIRECTIONS_PRIOR,
    )
    dgp_dirs = EXPECTED_DIRECTIONS_DGP[cohort_name]
    dgp_conf, dgp_table = mb.score_intervention_conformity(
        X.iloc[te], interventions=INTERVENTIONS, expected_directions=dgp_dirs,
    )
    # Merge the per-intervention tables: prior_passes vs dgp_passes per intervention.
    merged = prior_table.rename(
        columns={
            "passed_directionality": "passes_prior",
            "expected_direction":    "expected_direction_prior",
        }
    ).copy()
    merged["passes_dgp"] = dgp_table["passed_directionality"].to_numpy()
    merged["expected_direction_dgp"] = [
        dgp_dirs[i] for i in merged["intervention"]
    ]
    # Responsive DGP conformity: passes only when the *sign* of mean_risk_change
    # matches the DGP-expected direction AND the magnitude exceeds a non-trivial
    # threshold. This closes the "constraint drove coefficient to zero so the
    # intervention has no effect and gets credit for non-response" loophole.
    deltas = merged["mean_risk_change"].to_numpy(dtype=float)
    expected_dgp = merged["expected_direction_dgp"].to_numpy(dtype=int)
    responsive = (np.sign(deltas) == np.sign(expected_dgp)) & (np.abs(deltas) >= RESPONSIVE_THRESHOLD)
    # Interventions with expected direction 0 don't count (none in our setup).
    scored = expected_dgp != 0
    responsive_dgp = float(responsive[scored].mean()) if scored.any() else float("nan")
    merged["responsive_dgp"] = responsive

    bootstrap_t0 = perf_counter()
    boot = bootstrap_conformity(
        mb, X.iloc[te], INTERVENTIONS,
        EXPECTED_DIRECTIONS_PRIOR, dgp_dirs,
        n_bootstrap=n_bootstrap,
        random_state=seed + 10_000,
        responsive_threshold=RESPONSIVE_THRESHOLD,
    )
    bootstrap_seconds = perf_counter() - bootstrap_t0
    (_, p_lo, p_hi) = boot["prior_conformity"]
    (_, d_lo, d_hi) = boot["dgp_conformity"]
    (_, r_lo, r_hi) = boot["responsive_dgp_conformity"]

    return ExperimentResult(
        cohort=cohort_name, variant=variant, seed=seed,
        auroc=auc, auprc=ap, brier=br,
        r2_mean=r2_mean, r2_per_state=r2_dict,
        prior_conformity=prior_conf, dgp_conformity=dgp_conf,
        responsive_dgp_conformity=responsive_dgp,
        prior_conformity_ci_lo=p_lo, prior_conformity_ci_hi=p_hi,
        dgp_conformity_ci_lo=d_lo, dgp_conformity_ci_hi=d_hi,
        responsive_dgp_ci_lo=r_lo, responsive_dgp_ci_hi=r_hi,
        per_intervention=merged,
        best_baseline_auroc=best_bl_auc,
        generate_seconds=generate_seconds,
        baseline_seconds=baseline_seconds,
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        bootstrap_seconds=bootstrap_seconds,
        total_seconds=perf_counter() - total_t0,
    )


def _agg(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MB-CNet × ICg-Bench multi-cohort sweep")
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=None,
        metavar="N",
        help=(
            "bootstrap resamples per experiment for conformity CIs "
            f"(default: env ICG_N_BOOTSTRAP or {_DEFAULT_N_BOOTSTRAP})"
        ),
    )
    parser.add_argument(
        "--cohorts",
        default=None,
        metavar="CSV",
        help="comma-separated subset of cohort names (default: all registered sweep cohorts)",
    )
    parser.add_argument(
        "--variants",
        default=None,
        metavar="CSV",
        help="comma-separated subset of v0_1,sign_constrained,sign_constrained_augmented",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="directory for benchmark CSV/JSON artifacts",
    )
    args = parser.parse_args(argv)
    n_boot = args.bootstrap
    if n_boot is None:
        n_boot = int(os.environ.get("ICG_N_BOOTSTRAP", str(_DEFAULT_N_BOOTSTRAP)))
    if n_boot < 0:
        raise SystemExit("--bootstrap must be >= 0")

    cohorts: tuple[str, ...]
    if args.cohorts:
        cohorts = tuple(s.strip() for s in args.cohorts.split(",") if s.strip())
        bad = [c for c in cohorts if c not in COHORTS]
        if bad:
            raise SystemExit(f"unknown cohort(s): {bad}. choices: {list(COHORTS)}")
    else:
        cohorts = COHORTS

    variants: tuple[str, ...]
    if args.variants:
        variants = tuple(s.strip() for s in args.variants.split(",") if s.strip())
        bad_v = [v for v in variants if v not in VARIANTS]
        if bad_v:
            raise SystemExit(f"unknown variant(s): {bad_v}. choices: {list(VARIANTS)}")
    else:
        variants = VARIANTS

    outdir = ensure_dir(args.outdir, fallback_prefix="icg-cast-bench-")
    print(
        f"n_bootstrap={n_boot}  cohorts={list(cohorts)}  variants={list(variants)}  "
        f"outdir={outdir}",
        flush=True,
    )

    results: list[ExperimentResult] = []
    for cohort in cohorts:
        for variant in variants:
            for seed in SEEDS:
                t = f"{cohort:>22} :: {variant:<28} seed={seed:>2}"
                try:
                    r = _run_one(cohort, variant, seed, n_bootstrap=n_boot)
                except Exception as e:
                    print(f"  {t}  FAILED: {type(e).__name__}: {e}", flush=True)
                    continue
                results.append(r)
                print(
                    f"  {t}  AUROC={r.auroc:.3f}  R^2={r.r2_mean:.3f}  "
                    f"prior={r.prior_conformity:.3f}  dgp={r.dgp_conformity:.3f}  "
                    f"resp_dgp={r.responsive_dgp_conformity:.3f}  "
                    f"gap={r.auroc-r.best_baseline_auroc:+.3f}",
                    flush=True,
                )

    if not results:
        print("no successful experiments; nothing to write.", flush=True)
        return

    per_seed = pd.DataFrame([
        {
            "cohort": r.cohort, "variant": r.variant, "seed": r.seed,
            "auroc": r.auroc, "auprc": r.auprc, "brier": r.brier,
            "best_baseline_auroc": r.best_baseline_auroc,
            "auroc_gap": r.auroc - r.best_baseline_auroc,
            "mean_recovery_r2": r.r2_mean,
            "prior_conformity":          r.prior_conformity,
            "dgp_conformity":            r.dgp_conformity,
            "responsive_dgp_conformity": r.responsive_dgp_conformity,
            "prior_conformity_ci_lo":    r.prior_conformity_ci_lo,
            "prior_conformity_ci_hi":    r.prior_conformity_ci_hi,
            "dgp_conformity_ci_lo":      r.dgp_conformity_ci_lo,
            "dgp_conformity_ci_hi":      r.dgp_conformity_ci_hi,
            "responsive_dgp_ci_lo":      r.responsive_dgp_ci_lo,
            "responsive_dgp_ci_hi":      r.responsive_dgp_ci_hi,
            "generate_seconds":          r.generate_seconds,
            "baseline_seconds":          r.baseline_seconds,
            "fit_seconds":               r.fit_seconds,
            "predict_seconds":           r.predict_seconds,
            "bootstrap_seconds":         r.bootstrap_seconds,
            "total_seconds":             r.total_seconds,
        }
        for r in results
    ])
    per_seed.to_csv(outdir / "per_seed.csv", index=False)

    summary_rows = []
    for (cohort, variant), sub in per_seed.groupby(["cohort", "variant"], sort=False):
        auc_m, auc_s = _agg(sub["auroc"])
        gap_m, gap_s = _agg(sub["auroc_gap"])
        r2_m, r2_s = _agg(sub["mean_recovery_r2"])
        pconf_m, pconf_s = _agg(sub["prior_conformity"])
        dconf_m, dconf_s = _agg(sub["dgp_conformity"])
        rconf_m, rconf_s = _agg(sub["responsive_dgp_conformity"])
        bl_m, bl_s = _agg(sub["best_baseline_auroc"])
        total_sec_m, total_sec_s = _agg(sub["total_seconds"])
        summary_rows.append({
            "cohort": cohort, "variant": variant,
            "n_seeds": int(len(sub)),
            "auroc_mean": auc_m, "auroc_sd": auc_s,
            "best_baseline_auroc_mean": bl_m, "best_baseline_auroc_sd": bl_s,
            "auroc_gap_mean": gap_m, "auroc_gap_sd": gap_s,
            "recovery_r2_mean": r2_m, "recovery_r2_sd": r2_s,
            "prior_conformity_mean":          pconf_m, "prior_conformity_sd":          pconf_s,
            "dgp_conformity_mean":            dconf_m, "dgp_conformity_sd":            dconf_s,
            "responsive_dgp_conformity_mean": rconf_m, "responsive_dgp_conformity_sd": rconf_s,
            "total_seconds_mean": total_sec_m, "total_seconds_sd": total_sec_s,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "summary.csv", index=False)

    print("\n=== aggregated (mean ± SD over seeds {7,13,31}) ===")
    show = summary.copy()
    for c, w in [
        ("auroc", "auroc"),
        ("recovery_r2", "recovery_r2"),
        ("prior_conformity",          "prior_conformity"),
        ("dgp_conformity",            "dgp_conformity"),
        ("responsive_dgp_conformity", "responsive_dgp_conformity"),
    ]:
        show[c] = show[f"{w}_mean"].map("{:.3f}".format) + " ± " + show[f"{w}_sd"].map("{:.3f}".format)
    show["auroc_gap"] = show["auroc_gap_mean"].map("{:+.3f}".format) + " ± " + show["auroc_gap_sd"].map("{:.3f}".format)
    print(show[["cohort", "variant", "auroc", "auroc_gap", "recovery_r2", "prior_conformity", "dgp_conformity", "responsive_dgp_conformity"]].to_string(index=False))

    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seeds": list(SEEDS),
                "cohorts_filter": list(cohorts),
                "variants_filter": list(variants),
                "cohorts_full_registry": list(COHORTS),
                "responsive_threshold": RESPONSIVE_THRESHOLD,
                "n_bootstrap": n_boot,
                "interventions": INTERVENTIONS,
                "expected_directions_prior": EXPECTED_DIRECTIONS_PRIOR,
                "expected_directions_dgp":   EXPECTED_DIRECTIONS_DGP,
                "summary": summary_rows,
            },
            f, indent=2,
        )

    for r in results:
        tag = f"{r.cohort}__{r.variant}__seed{r.seed}"
        r.per_intervention.to_csv(outdir / f"intervention_conformity__{tag}.csv", index=False)

    per_state_rows: list[dict[str, float | str | int]] = []
    for r in results:
        for unit, r2 in r.r2_per_state.items():
            per_state_rows.append({
                "cohort": r.cohort, "variant": r.variant, "seed": r.seed,
                "bottleneck_unit": unit, "recovery_r2": float(r2),
            })
    pd.DataFrame(per_state_rows).to_csv(outdir / "per_state_recovery_r2.csv", index=False)

    try:
        display_outdir = outdir.relative_to(REPO_ROOT)
    except ValueError:
        display_outdir = outdir
    print(f"\nartifacts written under {display_outdir}/  "
          f"(files: {len(list(outdir.iterdir()))})")


if __name__ == "__main__":
    main()
