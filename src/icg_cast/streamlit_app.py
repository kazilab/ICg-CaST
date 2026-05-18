"""Streamlit browser app for ICg-CaST workflows."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:  # Streamlit is an optional app dependency.
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - exercised only without the app extra.
    st = None  # type: ignore[assignment]

from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from icg_cast._branding import (
    BENCH_NAME,
    PROJECT_LONG_NAME,
    PROJECT_NAME,
    PROJECT_TAGLINE,
    VERSION,
)
from icg_cast.benchmark import dgp as _dgp
from icg_cast.benchmark import generate
from icg_cast.benchmark.conformity_bootstrap import (
    responsive_conformity_from_table,
    responsive_passes_from_table,
)
from icg_cast.biology.biological_risk_equation import biological_risk_equation
from icg_cast.bottleneck import DEFAULT_BOTTLENECK_UNITS
from icg_cast.cli import (
    _make_variant,
    _omics_feature_columns,
    _write_model_card,
)
from icg_cast.config import SimConfig
from icg_cast.graph import write_theory_graph
from icg_cast.interventions import (
    EXPECTED_DIRECTIONS_PRIOR as _EXPECTED_DIRECTIONS_PRIOR,
)
from icg_cast.interventions import (
    INTERVENTIONS as _INTERVENTIONS,
)
from icg_cast.interventions import (
    dgp_directions as _dgp_directions,
)
from icg_cast.interventions import (
    risk_function_directions as _risk_function_directions,
)
from icg_cast.io import ensure_dir, write_cohort, write_simulation_metadata
from icg_cast.models import biological_coherence_summary, evaluate_bundle, train_baselines
from icg_cast.oracle.reference_risk_oracle import reference_risk_oracle
from icg_cast.plotting import plot_metrics, plot_trajectories
from icg_cast.simulator import simulate_cohort

APP_OUTPUT_ROOT = Path("outputs") / "streamlit"


def _require_streamlit() -> None:
    if st is None:
        raise RuntimeError(
            "Streamlit is not installed. Install the app extra with "
            '`python -m pip install -e ".[app]"` and rerun `streamlit run streamlit_app.py`.'
        )


def _run_name(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    return safe or _run_name("run")


def _output_dir(run_name: str) -> Path:
    return ensure_dir(APP_OUTPUT_ROOT / _safe_name(run_name))


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def _mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "text/csv"
    if suffix == ".json":
        return "application/json"
    if suffix in {".md", ".txt"}:
        return "text/plain"
    if suffix == ".png":
        return "image/png"
    if suffix == ".joblib":
        return "application/octet-stream"
    return "application/octet-stream"


def _human_size(path: Path) -> str:
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _numeric_summary(df: pd.DataFrame) -> dict[str, float]:
    summary: dict[str, float] = {"n_rows": float(len(df)), "n_columns": float(len(df.columns))}
    if "future_cancer_transition_event" in df:
        summary["event_rate"] = float(df["future_cancer_transition_event"].mean())
    if "future_event_probability" in df:
        summary["mean_future_event_probability"] = float(df["future_event_probability"].mean())
    if "state_final_latent_risk" in df:
        summary["mean_final_latent_risk"] = float(df["state_final_latent_risk"].mean())
    return summary


def _simulate(
    *,
    n: int,
    months: int,
    seed: int,
    event_hazard_scale: float,
    coefficient_mode: str,
    coefficient_seed: int | None,
    outdir: Path,
    plots: bool,
) -> dict[str, Any]:
    outdir = ensure_dir(outdir)
    cfg = SimConfig(
        n=n,
        months=months,
        seed=seed,
        outdir=outdir,
        event_hazard_scale=event_hazard_scale,
        coefficient_mode=coefficient_mode,  # type: ignore[arg-type]
        coefficient_seed=coefficient_seed,
    )
    cohort, trajectories = simulate_cohort(cfg)
    cohort_path = write_cohort(cohort, outdir)
    metadata_path = write_simulation_metadata(
        cfg,
        outdir,
        extra={"n_trajectories_retained": len(trajectories), "synthetic_only": True},
    )
    trajectory_plot = plot_trajectories(trajectories, outdir) if plots else None
    return {
        "outdir": outdir,
        "config": asdict(cfg),
        "cohort": cohort,
        "paths": {
            "cohort": cohort_path,
            "metadata": metadata_path,
            "trajectory_plot": trajectory_plot,
        },
        "summary": _numeric_summary(cohort),
    }


def _train(
    *,
    cohort: pd.DataFrame,
    outdir: Path,
    seed: int,
    target: str,
    test_size: float,
    plots: bool,
) -> dict[str, Any]:
    outdir = ensure_dir(outdir)
    metrics, importance, counterfactual, bundle = train_baselines(
        cohort,
        seed=seed,
        target=target,
        test_size=test_size,
    )
    coherence = biological_coherence_summary(counterfactual)

    paths = {
        "metrics": outdir / "model_metrics.csv",
        "importance": outdir / "permutation_importance.csv",
        "counterfactual": outdir / "counterfactual_tests.csv",
        "coherence": outdir / "biological_coherence.csv",
        "bundle": outdir / "model_bundle.joblib",
    }
    metrics.to_csv(paths["metrics"], index=False)
    importance.to_csv(paths["importance"], index=False)
    counterfactual.to_csv(paths["counterfactual"], index=False)
    coherence.to_csv(paths["coherence"], index=False)
    joblib.dump(bundle, paths["bundle"])
    paths["model_card"] = _write_model_card(outdir, bundle, metrics, coherence)
    paths["metrics_plot"] = plot_metrics(metrics, outdir) if plots else None

    return {
        "metrics": metrics,
        "importance": importance,
        "counterfactual": counterfactual,
        "coherence": coherence,
        "bundle": bundle,
        "paths": paths,
    }


def _evaluate(
    *,
    cohort: pd.DataFrame,
    bundle: dict[str, Any],
    outdir: Path,
    full_cohort: bool,
) -> dict[str, Any]:
    outdir = ensure_dir(outdir)
    metrics, calibration, counterfactual, coherence = evaluate_bundle(
        cohort,
        bundle,
        use_heldout=not full_cohort,
    )
    paths = {
        "metrics": outdir / "evaluation_metrics.csv",
        "calibration": outdir / "calibration_metrics.csv",
        "counterfactual": outdir / "evaluation_counterfactual_tests.csv",
        "coherence": outdir / "evaluation_biological_coherence.csv",
    }
    metrics.to_csv(paths["metrics"], index=False)
    calibration.to_csv(paths["calibration"], index=False)
    counterfactual.to_csv(paths["counterfactual"], index=False)
    coherence.to_csv(paths["coherence"], index=False)

    return {
        "metrics": metrics,
        "calibration": calibration,
        "counterfactual": counterfactual,
        "coherence": coherence,
        "paths": paths,
    }


def _graph(outdir: Path) -> dict[str, Path]:
    outdir = ensure_dir(outdir)
    return write_theory_graph(outdir)


def _demo(
    *,
    n: int,
    months: int,
    seed: int,
    event_hazard_scale: float,
    coefficient_mode: str,
    coefficient_seed: int | None,
    target: str,
    test_size: float,
    outdir: Path,
    plots: bool,
) -> dict[str, Any]:
    outdir = ensure_dir(outdir)
    sim = _simulate(
        n=n,
        months=months,
        seed=seed,
        event_hazard_scale=event_hazard_scale,
        coefficient_mode=coefficient_mode,
        coefficient_seed=coefficient_seed,
        outdir=outdir,
        plots=plots,
    )
    train = _train(
        cohort=sim["cohort"],
        outdir=outdir,
        seed=seed,
        target=target,
        test_size=test_size,
        plots=plots,
    )
    eval_result = _evaluate(
        cohort=sim["cohort"],
        bundle=train["bundle"],
        outdir=outdir,
        full_cohort=False,
    )
    graph_paths = _graph(outdir)

    manifest = {
        "app": "streamlit",
        "package": PROJECT_NAME,
        "package_version": VERSION,
        "synthetic_only": True,
        "steps": ["simulate", "train", "evaluate", "graph"],
        "parameters": {
            "n": n,
            "months": months,
            "seed": seed,
            "event_hazard_scale": event_hazard_scale,
            "coefficient_mode": coefficient_mode,
            "coefficient_seed": coefficient_seed,
            "target": target,
            "test_size": test_size,
            "plots": plots,
        },
        "outputs": {path.name: str(path) for path in outdir.iterdir() if path.is_file()},
    }
    manifest_path = _write_json(outdir / "streamlit_run_manifest.json", manifest)
    return {
        "outdir": outdir,
        "simulate": sim,
        "train": train,
        "evaluate": eval_result,
        "graph": graph_paths,
        "manifest": manifest_path,
    }


def _benchmark(
    *,
    cohort_name: str,
    variant_name: str,
    seed: int,
    n: int,
    months: int,
    test_size: float,
    responsive_threshold: float,
    outdir: Path,
) -> dict[str, Any]:
    outdir = ensure_dir(outdir)
    df, _ = generate(cohort_name, n=n, months=months, seed=seed)
    feature_cols = _omics_feature_columns(df)
    bottleneck_cols = [c for c in DEFAULT_BOTTLENECK_UNITS if c in df.columns]
    x = df[feature_cols].copy()
    states = df[bottleneck_cols].copy()
    y = df["future_cancer_transition_event"].astype(int).to_numpy()
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )

    mb = _make_variant(variant_name, feature_cols, bottleneck_cols, seed=seed)
    mb.fit(x.iloc[train_idx].join(states.iloc[train_idx]), y[train_idx])
    proba = mb.predict_proba(x.iloc[test_idx])[:, 1]

    recovery = mb.score_recovery(x.iloc[test_idx], states.iloc[test_idx])
    prior_conformity, prior_table = mb.score_intervention_conformity(
        x.iloc[test_idx],
        interventions=_INTERVENTIONS,
        expected_directions=_EXPECTED_DIRECTIONS_PRIOR,
        responsive_threshold=responsive_threshold,
    )
    dgp_dirs = _dgp_directions(cohort_name)
    dgp_conformity, dgp_table = mb.score_intervention_conformity(
        x.iloc[test_idx],
        interventions=_INTERVENTIONS,
        expected_directions=dgp_dirs,
        responsive_threshold=responsive_threshold,
    )
    oracle_dirs = _risk_function_directions(states.iloc[test_idx], _INTERVENTIONS, reference_risk_oracle)
    oracle_conformity, oracle_table = mb.score_intervention_conformity(
        x.iloc[test_idx],
        interventions=_INTERVENTIONS,
        expected_directions=oracle_dirs,
        responsive_threshold=responsive_threshold,
    )
    biology_dirs = _risk_function_directions(states.iloc[test_idx], _INTERVENTIONS, biological_risk_equation)
    biology_conformity, biology_table = mb.score_intervention_conformity(
        x.iloc[test_idx],
        interventions=_INTERVENTIONS,
        expected_directions=biology_dirs,
        responsive_threshold=responsive_threshold,
    )

    deltas = prior_table["mean_risk_change"].to_numpy(dtype=float)
    expected_dgp = np.array([dgp_dirs[i] for i in prior_table["intervention"]], dtype=int)
    responsive = responsive_conformity_from_table(
        deltas,
        expected_dgp,
        threshold=responsive_threshold,
    )
    summary = {
        "cohort": cohort_name,
        "variant": variant_name,
        "seed": seed,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "auroc": float(roc_auc_score(y[test_idx], proba)),
        "auprc": float(average_precision_score(y[test_idx], proba)),
        "brier": float(brier_score_loss(y[test_idx], proba)),
        "recovery_r2_mean": float(recovery["recovery_r2"].mean()),
        "prior_conformity": float(prior_conformity),
        "dgp_conformity": float(dgp_conformity),
        "oracle_conformity": float(oracle_conformity),
        "biology_conformity": float(biology_conformity),
        "responsive_dgp_conformity": float(responsive),
    }

    intervention = prior_table.rename(
        columns={
            "passed_directionality": "passes_prior",
            "expected_direction": "expected_direction_prior",
        }
    ).copy()
    intervention["passes_dgp"] = dgp_table["passed_directionality"].to_numpy()
    intervention["expected_direction_dgp"] = expected_dgp
    intervention["passes_oracle"] = oracle_table["passed_directionality"].to_numpy()
    intervention["expected_direction_oracle"] = [oracle_dirs[i] for i in intervention["intervention"]]
    intervention["passes_biology"] = biology_table["passed_directionality"].to_numpy()
    intervention["expected_direction_biology"] = [biology_dirs[i] for i in intervention["intervention"]]
    intervention["responsive_dgp"] = responsive_passes_from_table(
        deltas,
        expected_dgp,
        threshold=responsive_threshold,
    )

    cohort_path = outdir / f"{cohort_name}_{variant_name}_cohort.csv"
    summary_path = _write_json(outdir / "benchmark_summary.json", summary)
    recovery_path = outdir / "bottleneck_recovery.csv"
    intervention_path = outdir / "intervention_conformity.csv"
    df.to_csv(cohort_path, index=False)
    recovery.to_csv(recovery_path, index=False)
    intervention.to_csv(intervention_path, index=False)

    return {
        "outdir": outdir,
        "summary": summary,
        "cohort": df,
        "recovery": recovery,
        "intervention": intervention,
        "paths": {
            "cohort": cohort_path,
            "summary": summary_path,
            "recovery": recovery_path,
            "intervention": intervention_path,
        },
    }


def _show_dataframe(label: str, df: pd.DataFrame, *, rows: int = 200) -> None:
    st.subheader(label)
    shown = df.head(rows)
    st.dataframe(shown, use_container_width=True, hide_index=True)
    if len(df) > rows:
        st.caption(f"Showing first {rows:,} of {len(df):,} rows.")


def _show_metrics(summary: dict[str, Any]) -> None:
    numeric = {
        key: value
        for key, value in summary.items()
        if isinstance(value, int | float | np.integer | np.floating) and not isinstance(value, bool)
    }
    if not numeric:
        return
    cols = st.columns(min(4, len(numeric)))
    for index, (key, value) in enumerate(numeric.items()):
        label = key.replace("_", " ")
        formatted = f"{float(value):.3f}" if isinstance(value, float | np.floating) else f"{int(value):,}"
        cols[index % len(cols)].metric(label, formatted)


def _show_image(path: Path | None, caption: str) -> None:
    if path is not None and path.exists():
        st.image(str(path), caption=caption, use_container_width=True)


def _show_files(outdir: Path) -> None:
    files = sorted(path for path in outdir.iterdir() if path.is_file())
    if not files:
        return
    rows = [{"file": path.name, "size": _human_size(path)} for path in files]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    cols = st.columns(3)
    for index, path in enumerate(files):
        with cols[index % 3]:
            st.download_button(
                label=path.name,
                data=path.read_bytes(),
                file_name=path.name,
                mime=_mime_type(path),
                key=f"download-{outdir}-{path.name}",
            )


def _show_demo_result(result: dict[str, Any]) -> None:
    outdir = result["outdir"]
    st.success(f"Completed run in {outdir}")
    _show_metrics(result["simulate"]["summary"])
    _show_image(result["simulate"]["paths"]["trajectory_plot"], "Representative latent-risk trajectories")
    _show_image(result["train"]["paths"]["metrics_plot"], "Held-out AUROC by feature set")
    _show_dataframe("Training Metrics", result["train"]["metrics"])
    _show_dataframe("Evaluation Metrics", result["evaluate"]["metrics"])
    _show_dataframe("Biological Coherence", result["train"]["coherence"])
    with st.expander("Output files", expanded=True):
        _show_files(outdir)


def _show_simulation_result(result: dict[str, Any]) -> None:
    outdir = result["outdir"]
    st.success(f"Wrote simulation outputs to {outdir}")
    _show_metrics(result["summary"])
    _show_image(result["paths"]["trajectory_plot"], "Representative latent-risk trajectories")
    _show_dataframe("Cohort Preview", result["cohort"], rows=100)
    with st.expander("Output files", expanded=True):
        _show_files(outdir)


def _show_training_result(result: dict[str, Any], outdir: Path) -> None:
    st.success(f"Wrote model outputs to {outdir}")
    _show_image(result["train"]["paths"]["metrics_plot"], "Held-out AUROC by feature set")
    _show_dataframe("Training Metrics", result["train"]["metrics"])
    _show_dataframe("Permutation Importance", result["train"]["importance"], rows=100)
    if result.get("evaluate") is not None:
        _show_dataframe("Evaluation Metrics", result["evaluate"]["metrics"])
        _show_dataframe("Calibration Metrics", result["evaluate"]["calibration"])
    with st.expander("Output files", expanded=True):
        _show_files(outdir)


def _show_benchmark_result(result: dict[str, Any]) -> None:
    outdir = result["outdir"]
    st.success(f"Wrote benchmark outputs to {outdir}")
    _show_metrics(result["summary"])
    _show_dataframe("Intervention Conformity", result["intervention"])
    _show_dataframe("Bottleneck Recovery", result["recovery"], rows=100)
    with st.expander("Output files", expanded=True):
        _show_files(outdir)


def _demo_tab() -> None:
    if "demo_run_name" not in st.session_state:
        st.session_state.demo_run_name = _run_name("demo")

    with st.form("demo-form"):
        st.subheader("Demo Workflow")
        run_name = st.text_input("Run name", key="demo_run_name")
        col1, col2, col3 = st.columns(3)
        n = col1.number_input("Synthetic subjects", min_value=20, max_value=10000, value=120, step=20)
        months = col2.number_input("Months", min_value=1, max_value=240, value=72, step=6)
        seed = col3.number_input("Seed", min_value=0, max_value=1_000_000, value=7, step=1)
        col4, col5, col6 = st.columns(3)
        event_hazard_scale = col4.number_input(
            "Event hazard scale",
            min_value=0.0,
            max_value=1.0,
            value=0.020,
            step=0.001,
            format="%.3f",
        )
        coefficient_mode = col5.selectbox("Coefficient mode", ["point", "prior_sample"])
        coefficient_seed_value = col6.number_input(
            "Coefficient seed",
            min_value=0,
            max_value=1_000_000,
            value=int(seed),
            step=1,
            disabled=coefficient_mode == "point",
        )
        col7, col8, col9 = st.columns(3)
        target = col7.text_input("Target", value="future_cancer_transition_event")
        test_size = col8.number_input("Test size", min_value=0.10, max_value=0.80, value=0.30, step=0.05)
        plots = col9.checkbox("Render plots", value=True)
        submitted = st.form_submit_button("Run demo")

    if submitted:
        outdir = _output_dir(run_name)
        coefficient_seed = int(coefficient_seed_value) if coefficient_mode == "prior_sample" else None
        with st.spinner("Running simulation, training, evaluation, and graph export..."):
            try:
                st.session_state.demo_result = _demo(
                    n=int(n),
                    months=int(months),
                    seed=int(seed),
                    event_hazard_scale=float(event_hazard_scale),
                    coefficient_mode=str(coefficient_mode),
                    coefficient_seed=coefficient_seed,
                    target=target,
                    test_size=float(test_size),
                    outdir=outdir,
                    plots=plots,
                )
            except Exception as exc:  # pragma: no cover - Streamlit displays the exception.
                st.error(str(exc))
                st.stop()

    if "demo_result" in st.session_state:
        _show_demo_result(st.session_state.demo_result)


def _simulate_tab() -> None:
    if "sim_run_name" not in st.session_state:
        st.session_state.sim_run_name = _run_name("simulate")

    with st.form("simulate-form"):
        st.subheader("Simulate Cohort")
        run_name = st.text_input("Run name", key="sim_run_name")
        col1, col2, col3 = st.columns(3)
        n = col1.number_input("Synthetic subjects", min_value=20, max_value=100000, value=120, step=20)
        months = col2.number_input("Months", min_value=1, max_value=240, value=72, step=6)
        seed = col3.number_input("Seed", min_value=0, max_value=1_000_000, value=7, step=1)
        col4, col5, col6 = st.columns(3)
        event_hazard_scale = col4.number_input(
            "Event hazard scale",
            min_value=0.0,
            max_value=1.0,
            value=0.020,
            step=0.001,
            format="%.3f",
        )
        coefficient_mode = col5.selectbox("Coefficient mode", ["point", "prior_sample"], key="sim_coeff_mode")
        coefficient_seed_value = col6.number_input(
            "Coefficient seed",
            min_value=0,
            max_value=1_000_000,
            value=int(seed),
            step=1,
            disabled=coefficient_mode == "point",
            key="sim_coeff_seed",
        )
        plots = st.checkbox("Render plots", value=True, key="sim_plots")
        submitted = st.form_submit_button("Generate cohort")

    if submitted:
        outdir = _output_dir(run_name)
        coefficient_seed = int(coefficient_seed_value) if coefficient_mode == "prior_sample" else None
        with st.spinner("Simulating cohort..."):
            try:
                st.session_state.simulation_result = _simulate(
                    n=int(n),
                    months=int(months),
                    seed=int(seed),
                    event_hazard_scale=float(event_hazard_scale),
                    coefficient_mode=str(coefficient_mode),
                    coefficient_seed=coefficient_seed,
                    outdir=outdir,
                    plots=plots,
                )
            except Exception as exc:  # pragma: no cover - Streamlit displays the exception.
                st.error(str(exc))
                st.stop()

    if "simulation_result" in st.session_state:
        _show_simulation_result(st.session_state.simulation_result)


def _cohort_from_controls(uploaded_file: Any, cohort_path: str, outdir: Path) -> pd.DataFrame | None:
    if uploaded_file is not None:
        upload_path = outdir / "uploaded_cohort.csv"
        upload_path.write_bytes(uploaded_file.getbuffer())
        return pd.read_csv(upload_path)
    if cohort_path.strip():
        return pd.read_csv(Path(cohort_path).expanduser())
    return None


def _train_tab() -> None:
    if "train_run_name" not in st.session_state:
        st.session_state.train_run_name = _run_name("model")

    with st.form("train-form"):
        st.subheader("Train And Evaluate")
        run_name = st.text_input("Run name", key="train_run_name")
        uploaded = st.file_uploader("Cohort CSV", type=["csv"])
        cohort_path = st.text_input("Existing cohort path", value="")
        col1, col2, col3 = st.columns(3)
        seed = col1.number_input("Seed", min_value=0, max_value=1_000_000, value=7, step=1, key="train_seed")
        test_size = col2.number_input("Test size", min_value=0.10, max_value=0.80, value=0.30, step=0.05)
        plots = col3.checkbox("Render plots", value=True, key="train_plots")
        col4, col5 = st.columns(2)
        target = col4.text_input("Target", value="future_cancer_transition_event")
        evaluate_after_train = col5.checkbox("Evaluate after train", value=True)
        full_cohort = st.checkbox("Evaluate on full cohort", value=False, disabled=not evaluate_after_train)
        submitted = st.form_submit_button("Train model")

    if submitted:
        outdir = _output_dir(run_name)
        with st.spinner("Training baseline models..."):
            try:
                cohort = _cohort_from_controls(uploaded, cohort_path, outdir)
                if cohort is None:
                    st.error("Provide a cohort CSV or an existing cohort path.")
                    st.stop()
                train_result = _train(
                    cohort=cohort,
                    outdir=outdir,
                    seed=int(seed),
                    target=target,
                    test_size=float(test_size),
                    plots=plots,
                )
                eval_result = (
                    _evaluate(
                        cohort=cohort,
                        bundle=train_result["bundle"],
                        outdir=outdir,
                        full_cohort=full_cohort,
                    )
                    if evaluate_after_train
                    else None
                )
                st.session_state.training_result = {
                    "outdir": outdir,
                    "train": train_result,
                    "evaluate": eval_result,
                }
            except Exception as exc:  # pragma: no cover - Streamlit displays the exception.
                st.error(str(exc))
                st.stop()

    if "training_result" in st.session_state:
        _show_training_result(st.session_state.training_result, st.session_state.training_result["outdir"])


def _benchmark_tab() -> None:
    if "bench_run_name" not in st.session_state:
        st.session_state.bench_run_name = _run_name("benchmark")

    variants = _dgp.list_variants()
    with st.form("benchmark-form"):
        st.subheader(BENCH_NAME)
        run_name = st.text_input("Run name", key="bench_run_name")
        col1, col2 = st.columns(2)
        cohort_name = col1.selectbox("DGP cohort", variants)
        variant_name = col2.selectbox(
            "MB-CNet variant",
            ["v0_1", "sign_constrained", "sign_constrained_augmented"],
            index=1,
        )
        col3, col4, col5 = st.columns(3)
        n = col3.number_input("Synthetic subjects", min_value=40, max_value=20000, value=400, step=40)
        months = col4.number_input("Months", min_value=1, max_value=240, value=36, step=6)
        seed = col5.number_input("Seed", min_value=0, max_value=1_000_000, value=7, step=1)
        col6, col7 = st.columns(2)
        test_size = col6.number_input("Test size", min_value=0.10, max_value=0.80, value=0.30, step=0.05)
        responsive_threshold = col7.number_input(
            "Responsive threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.005,
            step=0.001,
            format="%.3f",
        )
        submitted = st.form_submit_button("Run benchmark")

    if submitted:
        outdir = _output_dir(run_name)
        with st.spinner("Running benchmark experiment..."):
            try:
                st.session_state.benchmark_result = _benchmark(
                    cohort_name=str(cohort_name),
                    variant_name=str(variant_name),
                    seed=int(seed),
                    n=int(n),
                    months=int(months),
                    test_size=float(test_size),
                    responsive_threshold=float(responsive_threshold),
                    outdir=outdir,
                )
            except Exception as exc:  # pragma: no cover - Streamlit displays the exception.
                st.error(str(exc))
                st.stop()

    if "benchmark_result" in st.session_state:
        _show_benchmark_result(st.session_state.benchmark_result)


def _render_file(path: Path) -> None:
    st.subheader(path.name)
    if path.suffix.lower() == ".csv":
        _show_dataframe("CSV", pd.read_csv(path), rows=500)
    elif path.suffix.lower() == ".json":
        st.json(json.loads(path.read_text(encoding="utf-8")))
    elif path.suffix.lower() in {".md", ".txt"}:
        st.code(path.read_text(encoding="utf-8"), language="markdown")
    elif path.suffix.lower() == ".png":
        st.image(str(path), use_container_width=True)
    else:
        st.info(f"{path.name} is available for download.")
    st.download_button(
        label=f"Download {path.name}",
        data=path.read_bytes(),
        file_name=path.name,
        mime=_mime_type(path),
        key=f"browser-download-{path}",
    )


def _outputs_tab() -> None:
    st.subheader("Outputs")
    root = Path(st.text_input("Output root", value=str(APP_OUTPUT_ROOT))).expanduser()
    if not root.exists():
        st.info(f"No outputs found at {root}")
        return

    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        st.info(f"No files found under {root}")
        return
    choices = [str(path.relative_to(root)) for path in files]
    selected = st.selectbox("File", choices)
    _render_file(root / selected)


def _header() -> None:
    st.set_page_config(page_title=f"{PROJECT_NAME} Browser App", layout="wide")
    st.title(PROJECT_NAME)
    st.caption(f"{PROJECT_LONG_NAME} | {PROJECT_TAGLINE} | v{VERSION}")


def main() -> None:
    _require_streamlit()
    _header()
    tabs = st.tabs(["Demo", "Simulate", "Train/Evaluate", "Benchmark", "Outputs"])
    with tabs[0]:
        _demo_tab()
    with tabs[1]:
        _simulate_tab()
    with tabs[2]:
        _train_tab()
    with tabs[3]:
        _benchmark_tab()
    with tabs[4]:
        _outputs_tab()


if __name__ == "__main__":
    main()
