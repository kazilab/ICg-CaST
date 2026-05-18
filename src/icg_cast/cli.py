"""Command-line interface for ICg-CaST.

This entry point exposes the core simulator, graph exporter, and ICg-Bench DGP
variants. It is intentionally argparse-only (no click / typer) to keep the
dependency footprint minimal.

Subcommands::

    icg-cast simulate --n 1200 --months 72 --seed 7 --outdir outputs/demo
    icg-cast train --cohort outputs/demo/synthetic_icg_cohort.csv --outdir outputs/demo
    icg-cast evaluate --cohort outputs/demo/synthetic_icg_cohort.csv --model outputs/demo/model_bundle.joblib
    icg-cast graph --outdir outputs/demo
    icg-cast make-demo --outdir outputs/demo
    icg-cast bench list                    # list registered DGP variants
    icg-cast bench info <variant>          # describe a single variant
    icg-cast bench run                     # run one (cohort, variant, seed) experiment
        --cohort linear_lowhet
        --variant sign_constrained_augmented
        --seed 7
        --n 1200 --months 72
    icg-cast bench audit                   # relax one stage-2 sign at a time; CSV optional
        --cohort linear_lowhet --variant sign_constrained --seed 7
    icg-cast bench sweep                   # delegates to scripts/bottleneck_proof_of_concept.py
    icg-cast bench plots                   # render manuscript figures from outputs/bottleneck_v0_5/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ._branding import (
    BENCH_NAME,
    CLI_NAME,
    COHORT_FILENAME,
    GRAPH_EDGES_JSON_FILENAME,
    GRAPH_GRAPHML_FILENAME,
    PROJECT_NAME,
)
from .benchmark import dgp as _dgp
from .benchmark import generate
from .benchmark.conformity_bootstrap import (
    responsive_conformity_from_table,
    responsive_passes_from_table,
)
from .biology.biological_risk_equation import biological_risk_equation
from .bottleneck import (
    DEFAULT_BOTTLENECK_UNITS,
    MechanismBottleneckClassifier,
    starter_kit_latent_risk,
)
from .calibration import (
    build_calibration_bundle,
    calibrated_registry_from_bundle,
    load_calibration_bundle,
)
from .coefficients import EVIDENCE_LEVELS
from .coefficients import registry as _coeff_registry
from .coefficients import save_registry as _save_coeff_registry
from .config import SimConfig
from .data_sources import calibration_provenance_payload
from .graph import write_theory_graph
from .interventions import (
    EXPECTED_DIRECTIONS_DGP_OVERRIDES,
    EXPECTED_DIRECTIONS_PRIOR,
    INTERVENTIONS,
    dgp_directions,
    risk_function_directions,
)
from .io import ensure_dir, write_cohort, write_simulation_metadata
from .models import biological_coherence_summary, evaluate_bundle, train_baselines
from .oracle.reference_risk_oracle import reference_risk_oracle
from .plotting import plot_metrics, plot_trajectories
from .simulator import simulate_cohort

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Backwards-compatible private aliases. Prefer importing the public names from
# ``icg_cast.interventions`` directly in new code.
_INTERVENTIONS = INTERVENTIONS
_EXPECTED_DIRECTIONS_PRIOR = EXPECTED_DIRECTIONS_PRIOR
_EXPECTED_DIRECTIONS_DGP_OVERRIDES = EXPECTED_DIRECTIONS_DGP_OVERRIDES
_dgp_directions = dgp_directions
_risk_function_directions = risk_function_directions


def _parse_csv_list(arg: str | None) -> list[str] | None:
    if arg is None:
        return None
    return [s.strip() for s in arg.split(",") if s.strip()]


def _make_variant(
    variant: str,
    feature_cols: list[str],
    bottleneck_cols: list[str],
    seed: int,
) -> MechanismBottleneckClassifier:
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
            augment_interventions=_INTERVENTIONS,
            augment_latent_risk_fn=starter_kit_latent_risk,
            augment_hazard_scale=0.020,
            augment_months=72,
            augment_samples_per_intervention=2,
            **kwargs,
        )
    raise ValueError(f"unknown MB-CNet variant: {variant!r}")


def _omics_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c.startswith(("tx_", "epi_", "sig_activity_", "kcc", "host_"))
        or c in ("dose", "mut_total_count")
    ]


# ----------------------------------------------------------------------
# core simulate / graph
# ----------------------------------------------------------------------


def _cmd_simulate(args: argparse.Namespace) -> int:
    outdir = ensure_dir(args.outdir)
    cfg = SimConfig(
        n=args.n,
        months=args.months,
        seed=args.seed,
        outdir=outdir,
        event_hazard_scale=args.event_hazard_scale,
        coefficient_mode=args.coefficient_mode,
        coefficient_seed=args.coefficient_seed,
        simulator_backend=args.simulator_backend,
    )
    calibration = (
        load_calibration_bundle(args.calibration) if args.calibration is not None else None
    )
    cohort, trajectories = simulate_cohort(cfg, calibration=calibration)
    cohort_path = write_cohort(cohort, outdir)
    metadata_path = write_simulation_metadata(
        cfg,
        outdir,
        extra={
            "n_trajectories_retained": len(trajectories),
            "synthetic_only": True,
            "calibration_sources": (
                sorted(calibration.provenance) if calibration is not None else []
            ),
        },
    )
    trajectory_plot = None
    if not getattr(args, "no_plots", False):
        trajectory_plot = plot_trajectories(trajectories, outdir)
    print(f"wrote cohort -> {cohort_path}")
    print(f"wrote metadata -> {metadata_path}")
    if trajectory_plot is not None:
        print(f"wrote trajectory plot -> {trajectory_plot}")
    return 0


def _cmd_graph(args: argparse.Namespace) -> int:
    calibration = (
        load_calibration_bundle(args.calibration)
        if getattr(args, "calibration", None) is not None
        else None
    )
    written = write_theory_graph(args.outdir, calibration=calibration)
    for fmt, path in written.items():
        print(f"wrote {fmt} -> {path}")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    outdir = ensure_dir(args.outdir)
    bundle = build_calibration_bundle(
        cosmic_path=args.cosmic,
        cosmic_signature_columns=_parse_csv_list(args.cosmic_signatures),
        cosmic_name_map=_parse_kv_pairs(args.cosmic_name_map),
        lincs_path=args.lincs,
        lincs_metadata_path=args.lincs_metadata,
        lincs_module_map=args.lincs_module_map,
        lincs_perturbagen_column=args.lincs_perturbagen_column,
        lincs_gene_column=args.lincs_gene_column,
        lincs_score_column=args.lincs_score_column,
        toxcast_path=args.toxcast,
        toxcast_mapping=args.toxcast_mapping,
        toxcast_chemical_column=args.toxcast_chemical_column,
        toxcast_assay_column=args.toxcast_assay_column,
        toxcast_hitcall_column=args.toxcast_hitcall_column,
        aopwiki_path=args.aopwiki,
        aopdb_path=args.aopdb,
        aopdb_table=args.aopdb_table,
    )
    bundle_path = outdir / "calibration_bundle.json"
    bundle.save(bundle_path)
    provenance_path = outdir / "calibration_provenance.json"
    coefficient_summary = None
    coefficient_path = None
    if args.apply_coefficients:
        calibrated, coefficient_summary = calibrated_registry_from_bundle(
            _coeff_registry(),
            bundle,
        )
        coefficient_path = _save_coeff_registry(
            calibrated,
            outdir / "calibrated_coefficients.yaml",
        )
    provenance_payload = calibration_provenance_payload(
        bundle.provenance,
        coefficient_updates=coefficient_summary,
    )
    provenance_path.write_text(
        json.dumps(provenance_payload, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "calibration_sources": sorted(bundle.provenance),
        "signature_profiles": list((bundle.signature_profiles or {}).keys()),
        "archetype_kcc": list((bundle.archetype_kcc or {}).keys()),
        "transcript_module_priors_rows": len(bundle.transcript_module_priors or []),
        "graph_edges": len(bundle.graph_edges or []),
        "graph_nodes_with_attributes": len(bundle.graph_node_attributes or {}),
        "coefficient_updates": None if coefficient_summary is None else {
            "n_updates": coefficient_summary["n_updates"],
            "e1_e3_before": coefficient_summary["e1_e3_before"],
            "e1_e3_after": coefficient_summary["e1_e3_after"],
            "evidence_upgrade_count": coefficient_summary["evidence_upgrade_count"],
        },
    }
    print(json.dumps(summary, indent=2))
    print(f"wrote calibration bundle -> {bundle_path}")
    print(f"wrote calibration provenance -> {provenance_path}")
    if coefficient_path is not None:
        print(f"wrote calibrated coefficients -> {coefficient_path}")
    return 0


def _parse_kv_pairs(arg: str | None) -> dict[str, str] | None:
    if arg is None:
        return None
    out: dict[str, str] = {}
    for item in arg.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"expected key=value pairs, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _cmd_coeffs_list(args: argparse.Namespace) -> int:
    """List coefficient cards (optionally filtered by evidence level / prefix)."""
    r = _coeff_registry()
    cards = r.filter(evidence_level=args.evidence, prefix=args.prefix)
    rows: list[dict[str, object]] = []
    for c in cards:
        rows.append(
            {
                "name": c.name,
                "evidence": c.evidence_level,
                "default_value": (
                    list(c.default_value) if isinstance(c.default_value, tuple) else c.default_value
                ),
                "units": c.units,
                "prior_distribution": c.prior_distribution,
                "prior_params": dict(c.prior_params),
                "source": c.source,
            }
        )
    if not rows:
        print("(no coefficient cards match the filter)")
        return 0
    df = pd.DataFrame(rows)
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        with pd.option_context("display.max_rows", None, "display.max_colwidth", 80):
            print(df.to_string(index=False))
    print(
        f"\nshowing {len(rows)}/{len(r)} cards"
        + (f"  filter: evidence={args.evidence!r}" if args.evidence else "")
        + (f"  prefix={args.prefix!r}" if args.prefix else ""),
        file=sys.stderr,
    )
    return 0


def _cmd_coeffs_audit(args: argparse.Namespace) -> int:
    """Print a per-evidence-level summary of the coefficient registry."""
    r = _coeff_registry()
    rows = []
    for level in EVIDENCE_LEVELS:
        cards = r.filter(evidence_level=level)
        rows.append({"evidence": level, "n_cards": len(cards)})
    total = len(r)
    rows.append({"evidence": "total", "n_cards": total})
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nregistry schema_version: {r.schema_version}", file=sys.stderr)
    return 0


def _cmd_coeffs_sensitivity(args: argparse.Namespace) -> int:
    """Run the coefficient sensitivity audit."""
    from icg_cast.audit.coefficient_sensitivity import run_coefficient_sensitivity_audit

    cfg = SimConfig(
        n=args.n,
        months=args.months,
        seed=args.seed,
        outdir=args.outdir,
    )
    names = _parse_csv_list(args.coefficients)
    table = run_coefficient_sensitivity_audit(
        simulate_cohort,
        cfg,
        coefficients_to_test=names,
        output_dir=args.outdir,
        n_samples=args.n,
    )
    print(f"wrote sensitivity rows -> {Path(args.outdir) / 'coefficient_sensitivity.csv'}")
    print(f"wrote sensitivity heatmap -> {Path(args.outdir) / 'coefficient_sensitivity_heatmap.png'}")
    print(f"wrote coefficient flags -> {Path(args.outdir) / 'coefficient_flags.csv'}")
    print(f"audited {table['coefficient'].nunique() if len(table) else 0} coefficients")
    return 0


def _write_model_card(
    outdir: Path,
    bundle: dict[str, object],
    metrics: pd.DataFrame,
    coherence: pd.DataFrame | None = None,
) -> Path:
    best = metrics.iloc[0].to_dict() if len(metrics) else {}
    coherence_score = None
    if coherence is not None and len(coherence):
        coherence_score = coherence["biological_coherence_score"].iloc[0]
    lines = [
        f"# {PROJECT_NAME} Baseline Model Card",
        "",
        "## Intended Use",
        "Synthetic theory-development experiments only.",
        "",
        "## Prohibited Use",
        "Do not use this model for clinical, individual-risk, environmental safety, or regulatory decisions.",
        "",
        "## Training Data",
        f"Synthetic {PROJECT_NAME} cohort generated under explicit simulation assumptions.",
        "",
        "## Model",
        f"- model: {bundle.get('model_name')}",
        f"- feature set: {bundle.get('feature_set')}",
        f"- number of features: {len(bundle.get('feature_columns', []))}",
        f"- seed: {bundle.get('seed')}",
        f"- package version: {bundle.get('package_version')}",
        "",
        "## Best Held-Out Metrics From Training",
        f"- ROC AUC: {best.get('roc_auc', float('nan'))}",
        f"- average precision: {best.get('average_precision', float('nan'))}",
        f"- Brier score: {best.get('brier_score', float('nan'))}",
    ]
    if coherence_score is not None:
        lines.extend(["", "## Counterfactual Coherence", f"- biological coherence score: {coherence_score}"])
    path = outdir / "model_card.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _cmd_train(args: argparse.Namespace) -> int:
    import joblib

    outdir = ensure_dir(args.outdir)
    cohort = pd.read_csv(args.cohort)
    metrics, importance, counterfactual, bundle = train_baselines(
        cohort,
        seed=args.seed,
        target=args.target,
        test_size=args.test_size,
    )

    metrics_path = outdir / "model_metrics.csv"
    importance_path = outdir / "permutation_importance.csv"
    counterfactual_path = outdir / "counterfactual_tests.csv"
    bundle_path = outdir / "model_bundle.joblib"
    coherence = biological_coherence_summary(counterfactual)
    coherence_path = outdir / "biological_coherence.csv"

    metrics.to_csv(metrics_path, index=False)
    importance.to_csv(importance_path, index=False)
    counterfactual.to_csv(counterfactual_path, index=False)
    coherence.to_csv(coherence_path, index=False)
    joblib.dump(bundle, bundle_path)
    card_path = _write_model_card(outdir, bundle, metrics, coherence)
    plot_path = None
    if not args.no_plots:
        plot_path = plot_metrics(metrics, outdir)

    print(f"wrote metrics -> {metrics_path}")
    print(f"wrote importance -> {importance_path}")
    print(f"wrote counterfactuals -> {counterfactual_path}")
    print(f"wrote biological coherence -> {coherence_path}")
    print(f"wrote model bundle -> {bundle_path}")
    print(f"wrote model card -> {card_path}")
    if plot_path is not None:
        print(f"wrote plot -> {plot_path}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    import joblib

    outdir = ensure_dir(args.outdir)
    cohort = pd.read_csv(args.cohort)
    bundle = joblib.load(args.model)
    metrics, calibration, counterfactual, coherence = evaluate_bundle(
        cohort,
        bundle,
        use_heldout=not args.full_cohort,
    )

    metrics_path = outdir / "evaluation_metrics.csv"
    calibration_path = outdir / "calibration_metrics.csv"
    counterfactual_path = outdir / "counterfactual_tests.csv"
    coherence_path = outdir / "biological_coherence.csv"
    metrics.to_csv(metrics_path, index=False)
    calibration.to_csv(calibration_path, index=False)
    counterfactual.to_csv(counterfactual_path, index=False)
    coherence.to_csv(coherence_path, index=False)
    card_path = _write_model_card(outdir, bundle, metrics, coherence)

    print(f"wrote evaluation metrics -> {metrics_path}")
    print(f"wrote calibration -> {calibration_path}")
    print(f"wrote counterfactuals -> {counterfactual_path}")
    print(f"wrote biological coherence -> {coherence_path}")
    print(f"wrote model card -> {card_path}")
    return 0


def _cmd_make_demo(args: argparse.Namespace) -> int:
    """Run the complete reproducible demo workflow in one command."""
    outdir = ensure_dir(args.outdir, fallback_prefix="icg-cast-demo-")
    simulate_args = argparse.Namespace(
        n=args.n,
        months=args.months,
        seed=args.seed,
        event_hazard_scale=args.event_hazard_scale,
        coefficient_mode=args.coefficient_mode,
        coefficient_seed=args.coefficient_seed,
        simulator_backend=args.simulator_backend,
        outdir=outdir,
        no_plots=args.no_plots,
        calibration=None,
    )
    train_args = argparse.Namespace(
        cohort=outdir / COHORT_FILENAME,
        outdir=outdir,
        seed=args.seed,
        target=args.target,
        test_size=args.test_size,
        no_plots=args.no_plots,
    )
    evaluate_args = argparse.Namespace(
        cohort=outdir / COHORT_FILENAME,
        model=outdir / "model_bundle.joblib",
        outdir=outdir,
        full_cohort=False,
    )
    graph_args = argparse.Namespace(outdir=outdir)

    print("[make-demo] simulate")
    _cmd_simulate(simulate_args)
    print("[make-demo] train")
    _cmd_train(train_args)
    print("[make-demo] evaluate")
    _cmd_evaluate(evaluate_args)
    print("[make-demo] graph")
    _cmd_graph(graph_args)

    expected_outputs = [
        COHORT_FILENAME,
        "simulation_metadata.json",
        "model_metrics.csv",
        "permutation_importance.csv",
        "model_bundle.joblib",
        "model_card.md",
        "evaluation_metrics.csv",
        "calibration_metrics.csv",
        "counterfactual_tests.csv",
        "biological_coherence.csv",
        GRAPH_GRAPHML_FILENAME,
        GRAPH_EDGES_JSON_FILENAME,
    ]
    if not args.no_plots:
        expected_outputs.extend(["example_state_trajectories.png", "modality_auc.png"])
    manifest = {
        "command": f"{CLI_NAME} make-demo",
        "synthetic_only": True,
        "steps": ["simulate", "train", "evaluate", "graph", "plots"],
        "parameters": {
            "n": args.n,
            "months": args.months,
            "seed": args.seed,
            "event_hazard_scale": args.event_hazard_scale,
            "coefficient_mode": args.coefficient_mode,
            "coefficient_seed": args.coefficient_seed,
            "simulator_backend": args.simulator_backend,
            "target": args.target,
            "test_size": args.test_size,
            "plots": not args.no_plots,
        },
        "outputs": {name: str(outdir / name) for name in expected_outputs if (outdir / name).exists()},
    }
    manifest_path = outdir / "demo_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[make-demo] wrote manifest -> {manifest_path}")
    return 0


# ----------------------------------------------------------------------
# bench list / info
# ----------------------------------------------------------------------


def _cmd_bench_list(_args: argparse.Namespace) -> int:
    rows: list[dict[str, object]] = []
    for name in _dgp.list_variants():
        v = _dgp.load_variant(name)
        rows.append({
            "name": v.name,
            "coupling": v.coupling,
            "archetype_mode": v.archetype_mode,
            "host_heterogeneity": v.host_heterogeneity,
            "observability": v.observability,
            "hash": v.hash()[:10],
        })
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


def _cmd_bench_info(args: argparse.Namespace) -> int:
    v = _dgp.load_variant(args.name)
    print(json.dumps(asdict(v), indent=2))
    print()
    print(f"hash (sha256, first 10): {v.hash()[:10]}")
    print("DGP-expected intervention directions (vs prior):")
    for k, prior in _EXPECTED_DIRECTIONS_PRIOR.items():
        dgp_v = _dgp_directions(v.name)[k]
        flag = "  (flipped vs prior)" if dgp_v != prior else ""
        print(f"  {k:<36} prior={prior:+d}  dgp={dgp_v:+d}{flag}")
    return 0


# ----------------------------------------------------------------------
# bench run (single experiment)
# ----------------------------------------------------------------------


def _cmd_bench_run(args: argparse.Namespace) -> int:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split

    total_t0 = perf_counter()
    generate_t0 = perf_counter()
    df, _ = generate(args.cohort, n=args.n, months=args.months, seed=args.seed)
    generate_seconds = perf_counter() - generate_t0
    feat = _omics_feature_columns(df)
    bcols = [c for c in DEFAULT_BOTTLENECK_UNITS if c in df.columns]
    X = df[feat].copy()
    S = df[bcols].copy()
    y = df["future_cancer_transition_event"].astype(int).to_numpy()
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=args.test_size, stratify=y, random_state=args.seed)

    mb = _make_variant(args.variant, feat, bcols, seed=args.seed)
    fit_t0 = perf_counter()
    mb.fit(X.iloc[tr].join(S.iloc[tr]), y[tr])
    fit_seconds = perf_counter() - fit_t0
    predict_t0 = perf_counter()
    proba = mb.predict_proba(X.iloc[te])[:, 1]
    predict_seconds = perf_counter() - predict_t0

    auroc = float(roc_auc_score(y[te], proba))
    auprc = float(average_precision_score(y[te], proba))
    brier = float(brier_score_loss(y[te], proba))

    rec = mb.score_recovery(X.iloc[te], S.iloc[te])
    r2_mean = float(rec["recovery_r2"].mean())

    prior_conf, prior_tab = mb.score_intervention_conformity(
        X.iloc[te], interventions=_INTERVENTIONS, expected_directions=_EXPECTED_DIRECTIONS_PRIOR,
        responsive_threshold=args.responsive_threshold,
    )
    dgp_dirs = _dgp_directions(args.cohort)
    dgp_conf, dgp_tab = mb.score_intervention_conformity(
        X.iloc[te], interventions=_INTERVENTIONS, expected_directions=dgp_dirs,
        responsive_threshold=args.responsive_threshold,
    )
    oracle_dirs = _risk_function_directions(S.iloc[te], _INTERVENTIONS, reference_risk_oracle)
    oracle_conf, oracle_tab = mb.score_intervention_conformity(
        X.iloc[te], interventions=_INTERVENTIONS, expected_directions=oracle_dirs,
        responsive_threshold=args.responsive_threshold,
    )
    biology_dirs = _risk_function_directions(S.iloc[te], _INTERVENTIONS, biological_risk_equation)
    biology_conf, biology_tab = mb.score_intervention_conformity(
        X.iloc[te], interventions=_INTERVENTIONS, expected_directions=biology_dirs,
        responsive_threshold=args.responsive_threshold,
    )
    deltas = prior_tab["mean_risk_change"].to_numpy(dtype=float)
    expected_dgp = np.array([dgp_dirs[i] for i in prior_tab["intervention"]], dtype=int)
    responsive_dgp = responsive_conformity_from_table(
        deltas,
        expected_dgp,
        threshold=args.responsive_threshold,
    )

    print(json.dumps({
        "cohort": args.cohort, "variant": args.variant, "seed": args.seed,
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "auroc": auroc, "auprc": auprc, "brier": brier,
        "recovery_r2_mean": r2_mean,
        "prior_conformity":          float(prior_conf),
        "dgp_conformity":            float(dgp_conf),
        "oracle_conformity":         float(oracle_conf),
        "biology_conformity":        float(biology_conf),
        "responsive_dgp_conformity": responsive_dgp,
        "wall_clock_seconds": {
            "generate": generate_seconds,
            "fit": fit_seconds,
            "predict": predict_seconds,
            "total": perf_counter() - total_t0,
        },
    }, indent=2))
    if args.write_intervention_csv is not None:
        merged = prior_tab.rename(columns={
            "passed_directionality": "passes_prior",
            "expected_direction":    "expected_direction_prior",
        }).copy()
        merged["passes_dgp"]               = dgp_tab["passed_directionality"].to_numpy()
        merged["expected_direction_dgp"]   = expected_dgp
        merged["passes_oracle"]            = oracle_tab["passed_directionality"].to_numpy()
        merged["expected_direction_oracle"] = [oracle_dirs[i] for i in merged["intervention"]]
        merged["passes_biology"]           = biology_tab["passed_directionality"].to_numpy()
        merged["expected_direction_biology"] = [biology_dirs[i] for i in merged["intervention"]]
        merged["responsive_dgp"]           = responsive_passes_from_table(
            deltas,
            expected_dgp,
            threshold=args.responsive_threshold,
        )
        merged.to_csv(args.write_intervention_csv, index=False)
        print(f"wrote intervention CSV -> {args.write_intervention_csv}", file=sys.stderr)
    return 0


def _cmd_bench_audit(args: argparse.Namespace) -> int:
    from sklearn.model_selection import train_test_split

    from icg_cast.audit import prior_sensitivity

    if args.variant not in ("sign_constrained", "sign_constrained_augmented"):
        print("audit requires --variant sign_constrained or sign_constrained_augmented", file=sys.stderr)
        return 2

    df, _ = generate(args.cohort, n=args.n, months=args.months, seed=args.seed)
    feat = _omics_feature_columns(df)
    bcols = [c for c in DEFAULT_BOTTLENECK_UNITS if c in df.columns]
    X = df[feat].copy()
    S = df[bcols].copy()
    y = df["future_cancer_transition_event"].astype(int).to_numpy()
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=args.test_size, stratify=y, random_state=args.seed)

    mb = _make_variant(args.variant, feat, bcols, seed=args.seed)
    mb.fit(X.iloc[tr].join(S.iloc[tr]), y[tr])

    dgp_dirs = _dgp_directions(args.cohort)
    table = prior_sensitivity(
        mb,
        X.iloc[tr].join(S.iloc[tr]),
        y[tr],
        X.iloc[te],
        y[te],
        _INTERVENTIONS,
        dgp_dirs,
        responsive_threshold=args.responsive_threshold,
    )
    if args.out is not None:
        table.to_csv(args.out, index=False)
        print(f"wrote {args.out}", file=sys.stderr)
    print(table.to_string(index=False))
    return 0


# ----------------------------------------------------------------------
# bench sweep (multi-seed sweep)
# ----------------------------------------------------------------------


def _cmd_bench_sweep(args: argparse.Namespace) -> int:
    """Delegate the full sweep to scripts/bottleneck_proof_of_concept.py.

    The script is the source of truth for the sweep configuration; this command
    is a discoverable entry point so users don't have to know its exact path.
    """
    script = _REPO_ROOT / "scripts" / "bottleneck_proof_of_concept.py"
    if not script.exists():
        print(f"sweep script not found at {script}", file=sys.stderr)
        return 2
    import subprocess
    outdir = ensure_dir(args.outdir, fallback_prefix="icg-cast-bench-")
    return subprocess.call([sys.executable, str(script), "--outdir", str(outdir)])


def _cmd_bench_plots(args: argparse.Namespace) -> int:
    script = _REPO_ROOT / "scripts" / "render_manuscript_plots.py"
    if not script.exists():
        print(f"plot script not found at {script}", file=sys.stderr)
        return 2
    import subprocess
    outdir = ensure_dir(args.outdir, fallback_prefix="icg-cast-figures-")
    return subprocess.call([
        sys.executable,
        str(script),
        "--inputdir",
        str(args.inputdir),
        "--outdir",
        str(outdir),
    ])


# ----------------------------------------------------------------------
# parser
# ----------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=CLI_NAME, description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    p_sim = sub.add_parser("simulate", help=f"generate a synthetic {PROJECT_NAME} cohort")
    p_sim.add_argument("--n", type=int, default=1200, help="number of synthetic subjects")
    p_sim.add_argument("--months", type=int, default=72, help="number of simulated months")
    p_sim.add_argument("--seed", type=int, default=7, help="random seed")
    p_sim.add_argument("--event-hazard-scale", type=float, default=0.020)
    p_sim.add_argument(
        "--coefficient-mode",
        choices=["point", "prior_sample"],
        default="point",
        help="use registry point values or one seedable prior sample per cohort",
    )
    p_sim.add_argument(
        "--coefficient-seed",
        type=int,
        default=None,
        help="seed for --coefficient-mode prior_sample (defaults to --seed)",
    )
    p_sim.add_argument(
        "--simulator-backend",
        choices=["python", "vectorized"],
        default="python",
        help="state-recurrence backend; vectorized is faster for large cohorts",
    )
    p_sim.add_argument("--outdir", type=Path, default=Path("outputs/demo"))
    p_sim.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help=f"path to a calibration bundle JSON from `{CLI_NAME} calibrate`",
    )
    p_sim.add_argument("--no-plots", action="store_true", help="skip example_state_trajectories.png")
    p_sim.set_defaults(func=_cmd_simulate)

    p_train = sub.add_parser("train", help="train baseline models on a synthetic cohort")
    p_train.add_argument("--cohort", type=Path, required=True, help=f"path to {COHORT_FILENAME}")
    p_train.add_argument("--outdir", type=Path, default=Path("outputs/demo"))
    p_train.add_argument("--seed", type=int, default=7)
    p_train.add_argument("--target", default="future_cancer_transition_event")
    p_train.add_argument("--test-size", type=float, default=0.30)
    p_train.add_argument("--no-plots", action="store_true", help="skip modality_auc.png")
    p_train.set_defaults(func=_cmd_train)

    p_eval = sub.add_parser("evaluate", help="evaluate a saved model bundle on a cohort")
    p_eval.add_argument("--cohort", type=Path, required=True, help=f"path to {COHORT_FILENAME}")
    p_eval.add_argument("--model", type=Path, required=True, help="path to model_bundle.joblib")
    p_eval.add_argument("--outdir", type=Path, default=Path("outputs/demo"))
    p_eval.add_argument("--full-cohort", action="store_true", help="evaluate on all rows instead of held-out rows")
    p_eval.set_defaults(func=_cmd_evaluate)

    p_graph = sub.add_parser("graph", help="export the default theory graph")
    p_graph.add_argument("--outdir", type=Path, default=Path("outputs/demo"))
    p_graph.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="path to a calibration bundle JSON that enriches the theory graph",
    )
    p_graph.set_defaults(func=_cmd_graph)

    p_cal = sub.add_parser(
        "calibrate",
        help="build a calibration bundle from local COSMIC / LINCS / ToxCast / AOP files",
    )
    p_cal.add_argument("--outdir", type=Path, default=Path("outputs/calibration"))
    p_cal.add_argument("--cosmic", type=Path, default=None, help="local 96-channel COSMIC SBS matrix")
    p_cal.add_argument(
        "--cosmic-signatures",
        type=str,
        default=None,
        help="comma-separated subset of COSMIC signature columns to load",
    )
    p_cal.add_argument(
        "--cosmic-name-map",
        type=str,
        default=None,
        help="rename pairs in the form 'SBS4=SBS4_like,SBS24=SBS24_like'",
    )
    p_cal.add_argument("--lincs", type=Path, default=None, help="local LINCS gene-level table")
    p_cal.add_argument("--lincs-metadata", type=Path, default=None)
    p_cal.add_argument(
        "--lincs-module-map",
        type=Path,
        default=None,
        help="CSV/TSV mapping gene -> transcriptomic module (required if --lincs given)",
    )
    p_cal.add_argument("--lincs-perturbagen-column", default="perturbagen")
    p_cal.add_argument("--lincs-gene-column", default="gene")
    p_cal.add_argument("--lincs-score-column", default="score")
    p_cal.add_argument("--toxcast", type=Path, default=None, help="local ToxCast summary table")
    p_cal.add_argument(
        "--toxcast-mapping",
        type=Path,
        default=None,
        help="CSV/TSV assay -> KCC mapping (required if --toxcast given)",
    )
    p_cal.add_argument("--toxcast-chemical-column", default="chemical_id")
    p_cal.add_argument("--toxcast-assay-column", default="assay")
    p_cal.add_argument("--toxcast-hitcall-column", default="hit_call")
    p_cal.add_argument("--aopwiki", type=Path, default=None, help="local AOP-Wiki edge-list export")
    p_cal.add_argument("--aopdb", type=Path, default=None, help="local AOP-DB node-attribute export")
    p_cal.add_argument("--aopdb-table", type=str, default=None)
    p_cal.add_argument(
        "--apply-coefficients",
        action="store_true",
        help="write calibrated_coefficients.yaml with evidence-level upgrades",
    )
    p_cal.set_defaults(func=_cmd_calibrate)

    coeffs = sub.add_parser(
        "coeffs",
        help="inspect the coefficient registry",
    )
    csub = coeffs.add_subparsers(dest="coeffs_cmd", required=True)
    p_coeffs_list = csub.add_parser("list", help="list coefficient cards")
    p_coeffs_list.add_argument(
        "--evidence",
        choices=list(EVIDENCE_LEVELS),
        default=None,
        help="filter to one evidence level (e.g. E5 for unsourced)",
    )
    p_coeffs_list.add_argument(
        "--prefix",
        default=None,
        help="filter to coefficient names with this dotted prefix",
    )
    p_coeffs_list.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    p_coeffs_list.set_defaults(func=_cmd_coeffs_list)
    p_coeffs_audit = csub.add_parser(
        "audit",
        help="summarise the registry by evidence level",
    )
    p_coeffs_audit.set_defaults(func=_cmd_coeffs_audit)
    p_coeffs_sensitivity = csub.add_parser(
        "sensitivity",
        help="run the 0.5x/1x/2x coefficient sensitivity audit",
    )
    p_coeffs_sensitivity.add_argument("--outdir", type=Path, default=Path("outputs/audit"))
    p_coeffs_sensitivity.add_argument("--n", type=int, default=120)
    p_coeffs_sensitivity.add_argument("--months", type=int, default=24)
    p_coeffs_sensitivity.add_argument("--seed", type=int, default=7)
    p_coeffs_sensitivity.add_argument(
        "--coefficients",
        type=str,
        default=None,
        help="optional comma-separated coefficient names for a focused audit",
    )
    p_coeffs_sensitivity.set_defaults(func=_cmd_coeffs_sensitivity)

    p_demo = sub.add_parser("make-demo", help="run simulate, train, evaluate, graph, and demo plots")
    p_demo.add_argument("--n", type=int, default=120, help="number of synthetic subjects")
    p_demo.add_argument("--months", type=int, default=72, help="number of simulated months")
    p_demo.add_argument("--seed", type=int, default=7, help="random seed")
    p_demo.add_argument("--event-hazard-scale", type=float, default=0.020)
    p_demo.add_argument(
        "--coefficient-mode",
        choices=["point", "prior_sample"],
        default="point",
        help="use registry point values or one seedable prior sample per cohort",
    )
    p_demo.add_argument(
        "--coefficient-seed",
        type=int,
        default=None,
        help="seed for --coefficient-mode prior_sample (defaults to --seed)",
    )
    p_demo.add_argument(
        "--simulator-backend",
        choices=["python", "vectorized"],
        default="python",
        help="state-recurrence backend; vectorized is faster for large cohorts",
    )
    p_demo.add_argument("--outdir", type=Path, default=Path("outputs/demo"))
    p_demo.add_argument("--target", default="future_cancer_transition_event")
    p_demo.add_argument("--test-size", type=float, default=0.30)
    p_demo.add_argument("--no-plots", action="store_true", help="skip PNG plots")
    p_demo.set_defaults(func=_cmd_make_demo)

    bench = sub.add_parser("bench", help=f"{BENCH_NAME} benchmark commands")
    bsub = bench.add_subparsers(dest="bench_cmd", required=True)

    p_list = bsub.add_parser("list", help="list registered DGP variants")
    p_list.set_defaults(func=_cmd_bench_list)

    p_info = bsub.add_parser("info", help="describe a single DGP variant")
    p_info.add_argument("name", choices=_dgp.list_variants())
    p_info.set_defaults(func=_cmd_bench_info)

    p_run = bsub.add_parser("run", help="run one (cohort, variant, seed) experiment")
    p_run.add_argument("--cohort", required=True, choices=_dgp.list_variants())
    p_run.add_argument("--variant", required=True,
                       choices=["v0_1", "sign_constrained", "sign_constrained_augmented"])
    p_run.add_argument("--seed", type=int, default=7)
    p_run.add_argument("--n", type=int, default=1200)
    p_run.add_argument("--months", type=int, default=72)
    p_run.add_argument("--test-size", type=float, default=0.30)
    p_run.add_argument("--responsive-threshold", type=float, default=0.005)
    p_run.add_argument("--write-intervention-csv", type=Path, default=None)
    p_run.set_defaults(func=_cmd_bench_run)

    p_sweep = bsub.add_parser("sweep", help="run the full multi-seed multi-cohort sweep")
    p_sweep.add_argument("--outdir", type=Path, default=_REPO_ROOT / "outputs" / "bottleneck_v0_5")
    p_sweep.set_defaults(func=_cmd_bench_sweep)

    p_plots = bsub.add_parser("plots", help="render manuscript figures from cached artifacts")
    p_plots.add_argument("--inputdir", type=Path, default=_REPO_ROOT / "outputs" / "bottleneck_v0_5")
    p_plots.add_argument("--outdir", type=Path, default=_REPO_ROOT / "outputs" / "figures")
    p_plots.set_defaults(func=_cmd_bench_plots)

    p_audit = bsub.add_parser(
        "audit",
        help="per-bottleneck prior-sensitivity audit (relax one sign at a time on stage 2)",
    )
    p_audit.add_argument("--cohort", required=True, choices=_dgp.list_variants())
    p_audit.add_argument(
        "--variant", required=True,
        choices=["sign_constrained", "sign_constrained_augmented"],
    )
    p_audit.add_argument("--seed", type=int, default=7)
    p_audit.add_argument("--n", type=int, default=1200)
    p_audit.add_argument("--months", type=int, default=72)
    p_audit.add_argument("--test-size", type=float, default=0.30)
    p_audit.add_argument("--responsive-threshold", type=float, default=0.005)
    p_audit.add_argument("--out", type=Path, default=None)
    p_audit.set_defaults(func=_cmd_bench_audit)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
