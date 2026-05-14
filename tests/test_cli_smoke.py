from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_module_cli_help_succeeds() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    result = subprocess.run(
        [sys.executable, "-m", "icg_cast", "--help"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0
    assert "icg-cast" in result.stdout
    assert "simulate" in result.stdout
    assert "graph" in result.stdout
    assert "make-demo" in result.stdout
    assert "bench" in result.stdout


def test_simulate_and_graph_cli_write_outputs(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    outdir = tmp_path / "demo"

    sim = subprocess.run(
        [
            sys.executable,
            "-m",
            "icg_cast",
            "simulate",
            "--n",
            "20",
            "--months",
            "6",
            "--seed",
            "1",
            "--coefficient-mode",
            "prior_sample",
            "--coefficient-seed",
            "99",
            "--outdir",
            str(outdir),
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )
    assert sim.returncode == 0, sim.stderr
    assert (outdir / "synthetic_icg_cohort.csv").exists()
    assert (outdir / "simulation_metadata.json").exists()
    assert (outdir / "example_state_trajectories.png").exists()
    metadata = json.loads((outdir / "simulation_metadata.json").read_text(encoding="utf-8"))
    assert metadata["coefficient_mode"] == "prior_sample"
    assert metadata["coefficient_seed"] == 99

    graph = subprocess.run(
        [sys.executable, "-m", "icg_cast", "graph", "--outdir", str(outdir)],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )
    assert graph.returncode == 0, graph.stderr
    assert (outdir / "icg_theory_graph.graphml").exists()
    assert (outdir / "icg_theory_graph_edges.json").exists()


def test_train_and_evaluate_cli_write_outputs(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    outdir = tmp_path / "workflow"

    sim = subprocess.run(
        [
            sys.executable,
            "-m",
            "icg_cast",
            "simulate",
            "--n",
            "80",
            "--months",
            "72",
            "--seed",
            "7",
            "--outdir",
            str(outdir),
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )
    assert sim.returncode == 0, sim.stderr

    train = subprocess.run(
        [
            sys.executable,
            "-m",
            "icg_cast",
            "train",
            "--cohort",
            str(outdir / "synthetic_icg_cohort.csv"),
            "--outdir",
            str(outdir),
            "--seed",
            "7",
            "--no-plots",
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )
    assert train.returncode == 0, train.stderr
    assert (outdir / "model_metrics.csv").exists()
    assert (outdir / "permutation_importance.csv").exists()
    assert (outdir / "model_bundle.joblib").exists()
    assert (outdir / "model_card.md").exists()

    evaluate = subprocess.run(
        [
            sys.executable,
            "-m",
            "icg_cast",
            "evaluate",
            "--cohort",
            str(outdir / "synthetic_icg_cohort.csv"),
            "--model",
            str(outdir / "model_bundle.joblib"),
            "--outdir",
            str(outdir),
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )
    assert evaluate.returncode == 0, evaluate.stderr
    assert (outdir / "evaluation_metrics.csv").exists()
    assert (outdir / "calibration_metrics.csv").exists()
    assert (outdir / "biological_coherence.csv").exists()


def test_calibrate_cli_builds_bundle_from_local_mock_files(tmp_path: Path) -> None:
    import numpy as np
    import pandas as pd

    from icg_cast.signatures import mutation_context_labels

    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")

    cosmic_path = tmp_path / "cosmic.csv"
    rng = np.random.default_rng(0)
    pd.DataFrame(
        {
            "context": mutation_context_labels(),
            "SBS4": rng.gamma(1.0, 0.3, size=96) + 0.01,
            "SBS24": rng.gamma(1.0, 0.3, size=96) + 0.01,
        }
    ).to_csv(cosmic_path, index=False)

    toxcast_path = tmp_path / "toxcast.csv"
    pd.DataFrame(
        [
            {"chemical_id": "ChemA", "assay": "Ames", "hit_call": 1},
            {"chemical_id": "ChemB", "assay": "ROS_oxidative", "hit_call": 1},
        ]
    ).to_csv(toxcast_path, index=False)
    mapping_path = tmp_path / "kcc_map.csv"
    pd.DataFrame(
        [
            {"assay": "Ames", "kcc_id": "KCC2"},
            {"assay": "ROS_oxidative", "kcc_id": "KCC5"},
        ]
    ).to_csv(mapping_path, index=False)

    aopwiki_path = tmp_path / "aopwiki.csv"
    pd.DataFrame(
        [{"source": "DNA_adducts", "target": "mutation_rate", "relationship": "increases"}]
    ).to_csv(aopwiki_path, index=False)

    outdir = tmp_path / "calibration"

    cal = subprocess.run(
        [
            sys.executable, "-m", "icg_cast", "calibrate",
            "--outdir", str(outdir),
            "--cosmic", str(cosmic_path),
            "--cosmic-name-map", "SBS4=SBS4_like,SBS24=SBS24_like",
            "--toxcast", str(toxcast_path),
            "--toxcast-mapping", str(mapping_path),
            "--aopwiki", str(aopwiki_path),
            "--apply-coefficients",
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )
    assert cal.returncode == 0, cal.stderr
    bundle_path = outdir / "calibration_bundle.json"
    assert bundle_path.exists()
    assert (outdir / "calibration_provenance.json").exists()
    assert (outdir / "calibrated_coefficients.yaml").exists()
    provenance = json.loads((outdir / "calibration_provenance.json").read_text(encoding="utf-8"))
    assert provenance["coefficient_updates"]["e1_e3_after"] > provenance["coefficient_updates"]["e1_e3_before"]

    sim_outdir = tmp_path / "sim_with_calibration"
    sim = subprocess.run(
        [
            sys.executable, "-m", "icg_cast", "simulate",
            "--n", "15", "--months", "6", "--seed", "1",
            "--outdir", str(sim_outdir),
            "--calibration", str(bundle_path),
            "--no-plots",
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )
    assert sim.returncode == 0, sim.stderr
    metadata = json.loads((sim_outdir / "simulation_metadata.json").read_text(encoding="utf-8"))
    assert "cosmic" in metadata.get("calibration_sources", [])
    assert "toxcast" in metadata.get("calibration_sources", [])


def test_make_demo_cli_writes_reproducible_demo_outputs(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    outdir = tmp_path / "make_demo"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "icg_cast",
            "make-demo",
            "--n",
            "80",
            "--months",
            "72",
            "--seed",
            "7",
            "--outdir",
            str(outdir),
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    expected = {
        "synthetic_icg_cohort.csv",
        "simulation_metadata.json",
        "model_metrics.csv",
        "permutation_importance.csv",
        "model_bundle.joblib",
        "evaluation_metrics.csv",
        "calibration_metrics.csv",
        "icg_theory_graph.graphml",
        "icg_theory_graph_edges.json",
        "example_state_trajectories.png",
        "modality_auc.png",
        "demo_manifest.json",
    }
    assert expected.issubset({p.name for p in outdir.iterdir()})
    manifest = json.loads((outdir / "demo_manifest.json").read_text(encoding="utf-8"))
    assert manifest["steps"] == ["simulate", "train", "evaluate", "graph", "plots"]
    assert manifest["parameters"]["seed"] == 7
    assert manifest["synthetic_only"] is True
