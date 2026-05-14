"""End-to-end demonstration of the Milestone 7 calibration prototype.

This example writes tiny synthetic mock files to a temporary directory, builds
an opt-in calibration bundle from them, and runs a small simulator cohort plus
a theory-graph export with the bundle applied. No real COSMIC, LINCS, ToxCast,
or AOP-Wiki data is downloaded or required.

Run from the repo root:

    python examples/run_calibration.py

To run the equivalent workflow with your own local files, see
``docs/calibration.md`` for the ``icg-cast calibrate`` CLI invocation.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from icg_cast import (
    SimConfig,
    build_calibration_bundle,
    build_theory_graph,
    load_calibration_bundle,
    simulate_cohort,
)
from icg_cast.graph import export_graph
from icg_cast.signatures import mutation_context_labels


def _write_mock_cosmic(path: Path) -> Path:
    rng = np.random.default_rng(0)
    pd.DataFrame(
        {
            "context": mutation_context_labels(),
            "SBS4": rng.gamma(1.0, 0.3, size=96) + 0.01,
            "SBS24": rng.gamma(1.0, 0.3, size=96) + 0.01,
            "SBS22": rng.gamma(1.0, 0.3, size=96) + 0.01,
        }
    ).to_csv(path, index=False)
    return path


def _write_mock_toxcast(path: Path, mapping_path: Path) -> tuple[Path, Path]:
    pd.DataFrame(
        [
            {"chemical_id": "MockChemA", "assay": "Ames", "hit_call": 1},
            {"chemical_id": "MockChemA", "assay": "MN_test", "hit_call": 1},
            {"chemical_id": "MockChemA", "assay": "ROS_oxidative", "hit_call": 0},
            {"chemical_id": "MockChemB", "assay": "ROS_oxidative", "hit_call": 1},
            {"chemical_id": "MockChemB", "assay": "Ames", "hit_call": 0},
        ]
    ).to_csv(path, index=False)
    pd.DataFrame(
        [
            {"assay": "Ames", "kcc_id": "KCC2"},
            {"assay": "MN_test", "kcc_id": "KCC2"},
            {"assay": "ROS_oxidative", "kcc_id": "KCC5"},
        ]
    ).to_csv(mapping_path, index=False)
    return path, mapping_path


def _write_mock_lincs(path: Path, map_path: Path) -> tuple[Path, Path]:
    pd.DataFrame(
        [
            {"perturbagen": "MockChemA", "gene": "TP53", "score": 1.2},
            {"perturbagen": "MockChemA", "gene": "MDM2", "score": -0.4},
            {"perturbagen": "MockChemB", "gene": "ESR1", "score": 0.9},
        ]
    ).to_csv(path, index=False)
    pd.DataFrame(
        [
            {"gene": "TP53", "module": "p53_checkpoint"},
            {"gene": "MDM2", "module": "p53_checkpoint"},
            {"gene": "ESR1", "module": "nuclear_receptor_program"},
        ]
    ).to_csv(map_path, index=False)
    return path, map_path


def _write_mock_aop(path: Path) -> Path:
    pd.DataFrame(
        [
            {"source": "DNA_adducts", "target": "mutation_rate", "relationship": "increases"},
            {"source": "ROS", "target": "lipid_peroxidation", "relationship": "increases"},
        ]
    ).to_csv(path, index=False)
    return path


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="icg_calibration_") as tmp:
        tmp_path = Path(tmp)
        cosmic = _write_mock_cosmic(tmp_path / "cosmic.csv")
        toxcast, mapping = _write_mock_toxcast(
            tmp_path / "toxcast.csv", tmp_path / "kcc_map.csv"
        )
        lincs, module_map = _write_mock_lincs(
            tmp_path / "lincs.csv", tmp_path / "module_map.csv"
        )
        aop = _write_mock_aop(tmp_path / "aopwiki.csv")

        bundle = build_calibration_bundle(
            cosmic_path=cosmic,
            cosmic_name_map={"SBS4": "SBS4_like", "SBS24": "SBS24_like", "SBS22": "SBS22_like"},
            lincs_path=lincs,
            lincs_module_map=module_map,
            toxcast_path=toxcast,
            toxcast_mapping=mapping,
            aopwiki_path=aop,
        )
        bundle_path = tmp_path / "calibration_bundle.json"
        bundle.save(bundle_path)

        roundtrip = load_calibration_bundle(bundle_path)

        cohort, _ = simulate_cohort(SimConfig(n=30, months=6, seed=11), calibration=roundtrip)
        graph = build_theory_graph(calibration=roundtrip)
        export_graph(graph, tmp_path)

        summary = {
            "calibration_sources": sorted(roundtrip.provenance),
            "signature_profiles_calibrated": sorted((roundtrip.signature_profiles or {}).keys()),
            "calibrated_archetypes": sorted((roundtrip.archetype_kcc or {}).keys()),
            "transcript_module_prior_rows": len(roundtrip.transcript_module_priors or []),
            "graph_edges_added_from_aop": len(roundtrip.graph_edges or []),
            "cohort_rows": int(len(cohort)),
            "cohort_archetypes": sorted(cohort["chemical_archetype"].unique()),
            "enriched_graph_has_lipid_peroxidation": "lipid_peroxidation" in graph.nodes,
        }
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
