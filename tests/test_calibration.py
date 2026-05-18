"""Tests for the public-data calibration prototype.

All tests use synthetic fixtures written into ``tmp_path``. No real COSMIC,
LINCS, ToxCast, AOP-Wiki, or AOP-DB files are downloaded or committed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from icg_cast import (
    KCC_NAMES,
    CalibrationBundle,
    SimConfig,
    build_calibration_bundle,
    build_theory_graph,
    load_calibration_bundle,
    make_signature_profiles,
    simulate_cohort,
)
from icg_cast.calibration import (
    calibrate_kcc_priors_from_toxcast,
    calibrate_signatures_from_cosmic,
    calibrate_transcript_modules_from_lincs,
    calibrated_registry_from_bundle,
    enrich_theory_graph,
)
from icg_cast.coefficients import load_registry, registry, save_registry, use_registry
from icg_cast.data_sources import (
    load_aopdb_export,
    load_aopwiki_export,
    load_cosmic_sbs_matrix,
    load_lincs_signatures,
    load_toxcast_summary,
)
from icg_cast.signatures import mutation_context_labels


def _write_cosmic_matrix(path: Path) -> Path:
    labels = mutation_context_labels()
    rng = np.random.default_rng(0)
    pd.DataFrame(
        {
            "context": labels,
            "SBS4": rng.gamma(1.0, 0.3, size=96) + 0.01,
            "SBS24": rng.gamma(1.0, 0.3, size=96) + 0.01,
            "SBS22": rng.gamma(1.0, 0.3, size=96) + 0.01,
        }
    ).to_csv(path, index=False)
    return path


def test_cosmic_calibrator_normalises_columns_and_renames(tmp_path: Path) -> None:
    cosmic_path = _write_cosmic_matrix(tmp_path / "cosmic.csv")
    bundle = load_cosmic_sbs_matrix(cosmic_path)

    labels, profiles = calibrate_signatures_from_cosmic(
        bundle, name_map={"SBS4": "SBS4_like", "SBS24": "SBS24_like"}
    )

    assert len(labels) == 96
    assert set(profiles) == {"SBS4_like", "SBS24_like", "SBS22"}
    for name, arr in profiles.items():
        assert arr.shape == (96,)
        assert (arr >= 0).all()
        assert np.isclose(arr.sum(), 1.0), f"signature {name} not normalised"


def test_cosmic_calibrator_rejects_non_cosmic_bundle(tmp_path: Path) -> None:
    path = tmp_path / "mock.csv"
    pd.DataFrame({"x": [1, 2, 3]}).to_csv(path, index=False)
    bundle = load_lincs_signatures(path)

    with pytest.raises(ValueError, match="COSMIC adapter bundle"):
        calibrate_signatures_from_cosmic(bundle)


def test_lincs_calibrator_aggregates_gene_scores_per_module(tmp_path: Path) -> None:
    lincs_path = tmp_path / "lincs.csv"
    pd.DataFrame(
        [
            {"perturbagen": "BPA", "gene": "TP53", "score": 1.2},
            {"perturbagen": "BPA", "gene": "MDM2", "score": -0.4},
            {"perturbagen": "BPA", "gene": "ESR1", "score": 0.9},
            {"perturbagen": "Aflatoxin", "gene": "TP53", "score": 1.8},
            {"perturbagen": "Aflatoxin", "gene": "BRCA1", "score": 0.5},
        ]
    ).to_csv(lincs_path, index=False)
    module_map_path = tmp_path / "module_map.csv"
    pd.DataFrame(
        [
            {"gene": "TP53", "module": "p53_checkpoint"},
            {"gene": "MDM2", "module": "p53_checkpoint"},
            {"gene": "ESR1", "module": "nuclear_receptor_program"},
            {"gene": "BRCA1", "module": "nucleotide_excision_repair"},
        ]
    ).to_csv(module_map_path, index=False)
    bundle = load_lincs_signatures(lincs_path)

    table = calibrate_transcript_modules_from_lincs(bundle, module_map_path)

    assert set(table.columns) == {"perturbagen", "module", "mean_score", "n_genes"}
    p53_bpa = table.query("perturbagen == 'BPA' and module == 'p53_checkpoint'")
    assert len(p53_bpa) == 1
    assert np.isclose(p53_bpa["mean_score"].iloc[0], (1.2 + -0.4) / 2)
    assert int(p53_bpa["n_genes"].iloc[0]) == 2


def test_toxcast_calibrator_builds_per_chemical_kcc_vectors(tmp_path: Path) -> None:
    toxcast_path = tmp_path / "toxcast.csv"
    pd.DataFrame(
        [
            {"chemical_id": "ChemA", "assay": "Ames", "hit_call": 1},
            {"chemical_id": "ChemA", "assay": "MN_test", "hit_call": 1},
            {"chemical_id": "ChemA", "assay": "ROS_oxidative", "hit_call": 0},
            {"chemical_id": "ChemB", "assay": "ROS_oxidative", "hit_call": 1},
            {"chemical_id": "ChemB", "assay": "Ames", "hit_call": 0},
        ]
    ).to_csv(toxcast_path, index=False)
    mapping_path = tmp_path / "kcc_map.csv"
    pd.DataFrame(
        [
            {"assay": "Ames", "kcc_id": "KCC2"},
            {"assay": "MN_test", "kcc_id": "KCC2"},
            {"assay": "ROS_oxidative", "kcc_id": "KCC5"},
        ]
    ).to_csv(mapping_path, index=False)
    bundle = load_toxcast_summary(toxcast_path)

    priors = calibrate_kcc_priors_from_toxcast(bundle, mapping_path)

    assert set(priors) == {"ChemA", "ChemB"}
    for vec in priors.values():
        assert len(vec) == len(KCC_NAMES)
        assert all(0.0 <= v <= 1.0 for v in vec)
    # ChemA: KCC2 = mean(1, 1) = 1.0; KCC5 = mean(0) = 0.0
    assert priors["ChemA"][1] == pytest.approx(1.0)
    assert priors["ChemA"][4] == pytest.approx(0.0)
    # ChemB: KCC2 = mean(0) = 0.0; KCC5 = 1.0
    assert priors["ChemB"][1] == pytest.approx(0.0)
    assert priors["ChemB"][4] == pytest.approx(1.0)


def test_aop_graph_enrichment_adds_edges_and_node_attributes(tmp_path: Path) -> None:
    aopwiki_path = tmp_path / "aopwiki.csv"
    pd.DataFrame(
        [
            {"source": "DNA_adducts", "target": "mutation_rate", "relationship": "increases"},
            {"source": "ROS", "target": "lipid_peroxidation", "relationship": "increases"},
        ]
    ).to_csv(aopwiki_path, index=False)
    aopdb_path = tmp_path / "aopdb.csv"
    pd.DataFrame(
        [
            {"node_id": "DNA_adducts", "tissue": "liver", "evidence": "high"},
            {"node_id": "lipid_peroxidation", "tissue": "membrane", "evidence": "moderate"},
        ]
    ).to_csv(aopdb_path, index=False)

    base = build_theory_graph()
    enriched = enrich_theory_graph(
        base,
        aopwiki=load_aopwiki_export(aopwiki_path),
        aopdb=load_aopdb_export(aopdb_path),
    )

    assert enriched.has_edge("DNA_adducts", "mutation_rate")
    assert enriched.has_edge("ROS", "lipid_peroxidation")
    assert enriched.nodes["DNA_adducts"]["tissue"] == "liver"
    assert enriched.nodes["lipid_peroxidation"]["node_type"] == "enriched"
    assert "provenance" in enriched.graph
    # base graph must not be mutated
    assert not base.has_edge("DNA_adducts", "mutation_rate")


def test_calibration_bundle_save_load_roundtrip(tmp_path: Path) -> None:
    bundle = CalibrationBundle(
        signature_labels=mutation_context_labels(),
        signature_profiles={"aging": [1.0 / 96] * 96},
        archetype_kcc={"chemX": [0.1] * 10},
        graph_edges=[{"source": "A", "target": "B"}],
        provenance={"cosmic": {"source_name": "COSMIC"}},
    )

    path = bundle.save(tmp_path / "calibration_bundle.json")
    loaded = load_calibration_bundle(path)

    assert loaded.signature_labels == bundle.signature_labels
    assert loaded.signature_profiles == bundle.signature_profiles
    assert loaded.archetype_kcc == bundle.archetype_kcc
    assert loaded.graph_edges == bundle.graph_edges
    assert loaded.provenance == bundle.provenance


def test_simulate_cohort_accepts_calibration_bundle(tmp_path: Path) -> None:
    cosmic_path = _write_cosmic_matrix(tmp_path / "cosmic.csv")
    toxcast_path = tmp_path / "toxcast.csv"
    pd.DataFrame(
        [
            {"chemical_id": "ChemA", "assay": "Ames", "hit_call": 1},
            {"chemical_id": "ChemA", "assay": "ROS_oxidative", "hit_call": 0},
            {"chemical_id": "ChemB", "assay": "Ames", "hit_call": 0},
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

    bundle = build_calibration_bundle(
        cosmic_path=cosmic_path,
        cosmic_name_map={"SBS4": "SBS4_like", "SBS24": "SBS24_like", "SBS22": "SBS22_like"},
        toxcast_path=toxcast_path,
        toxcast_mapping=mapping_path,
    )

    cfg = SimConfig(n=10, months=4, seed=11)
    cohort, _ = simulate_cohort(cfg, calibration=bundle)

    assert len(cohort) == 10
    assert set(cohort["chemical_archetype"].unique()).issubset({"ChemA", "ChemB"})


def test_build_theory_graph_with_calibration_merges_aop_edges(tmp_path: Path) -> None:
    aopwiki_path = tmp_path / "aopwiki.csv"
    pd.DataFrame(
        [{"source": "DNA_adducts", "target": "mutation_rate", "relationship": "increases"}]
    ).to_csv(aopwiki_path, index=False)
    bundle = build_calibration_bundle(aopwiki_path=aopwiki_path)

    graph = build_theory_graph(calibration=bundle)

    assert graph.has_edge("DNA_adducts", "mutation_rate")
    assert "aopwiki" in graph.graph["provenance"]


def test_make_signature_profiles_preserves_toy_keys_when_partial_calibration(tmp_path: Path) -> None:
    labels = mutation_context_labels()
    cal = CalibrationBundle(
        signature_labels=labels,
        signature_profiles={"SBS4_like": [1.0 / 96] * 96},
    )

    _, profiles = make_signature_profiles(calibration=cal)

    # calibrated key overrides
    assert np.isclose(profiles["SBS4_like"].sum(), 1.0)
    assert np.allclose(profiles["SBS4_like"], 1.0 / 96)
    # toy keys still present
    for key in ("aging", "SBS24_like", "SBS22_like", "oxidative_like"):
        assert key in profiles


def test_make_signature_profiles_reorders_toy_profiles_for_calibrated_labels() -> None:
    labels, base_profiles = make_signature_profiles()
    reordered_labels = list(reversed(labels))
    cal = CalibrationBundle(
        signature_labels=reordered_labels,
        signature_profiles={"SBS4_like": [1.0 / 96] * 96},
    )

    calibrated_labels, profiles = make_signature_profiles(calibration=cal)

    assert calibrated_labels == reordered_labels
    assert np.allclose(profiles["aging"], base_profiles["aging"][::-1])
    assert np.allclose(profiles["SBS22_like"], base_profiles["SBS22_like"][::-1])
    assert np.allclose(profiles["SBS4_like"], 1.0 / 96)


def test_make_signature_profiles_rejects_incompatible_calibrated_labels() -> None:
    labels = mutation_context_labels()
    bad_labels = labels.copy()
    bad_labels[0] = "not_a_valid_context"
    cal = CalibrationBundle(
        signature_labels=bad_labels,
        signature_profiles={"SBS4_like": [1.0 / 96] * 96},
    )

    with pytest.raises(ValueError, match="must be a permutation"):
        make_signature_profiles(calibration=cal)


def test_default_simulate_unchanged_without_calibration() -> None:
    cfg = SimConfig(n=20, months=6, seed=1)
    first, _ = simulate_cohort(cfg)
    second, _ = simulate_cohort(cfg, calibration=None)
    pd.testing.assert_frame_equal(first, second)


def test_apply_coefficients_upgrades_registry_and_changes_simulation(tmp_path: Path) -> None:
    toxcast_path = tmp_path / "toxcast.csv"
    pd.DataFrame(
        [
            {"chemical_id": "ChemA", "assay": "ER_proliferation", "hit_call": 1},
            {"chemical_id": "ChemA", "assay": "Ames", "hit_call": 1},
        ]
    ).to_csv(toxcast_path, index=False)
    mapping_path = tmp_path / "kcc_map.csv"
    pd.DataFrame(
        [
            {"assay": "ER_proliferation", "kcc_id": "KCC8"},
            {"assay": "Ames", "kcc_id": "KCC2"},
        ]
    ).to_csv(mapping_path, index=False)
    aopwiki_path = tmp_path / "aopwiki.csv"
    pd.DataFrame(
        [{"source": "KCC8", "target": "proliferation", "confidence": 0.9}]
    ).to_csv(aopwiki_path, index=False)

    bundle = build_calibration_bundle(
        toxcast_path=toxcast_path,
        toxcast_mapping=mapping_path,
        aopwiki_path=aopwiki_path,
    )
    base = registry()
    calibrated, summary = calibrated_registry_from_bundle(base, bundle)

    assert summary["e1_e3_after"] > summary["e1_e3_before"]
    assert calibrated.get("dynamics.proliferation.dose_kcc8_coupling") != pytest.approx(
        base.get("dynamics.proliferation.dose_kcc8_coupling")
    )
    assert "archetypes.chema.kcc" in calibrated

    path = save_registry(calibrated, tmp_path / "calibrated_coefficients.yaml")
    loaded = load_registry(path)
    assert loaded.get_vector("archetypes.chema.kcc")[7] == pytest.approx(1.0)

    cfg = SimConfig(n=40, months=12, seed=5, archetype_prior={"pah_tobacco_like": 1.0})
    point, _ = simulate_cohort(cfg)
    with use_registry(calibrated):
        changed, _ = simulate_cohort(cfg)
    assert float(changed["future_event_probability"].mean()) != pytest.approx(
        float(point["future_event_probability"].mean())
    )


def test_lincs_priors_flow_into_generate_omics_weights(tmp_path: Path) -> None:
    lincs_path = tmp_path / "lincs.csv"
    pd.DataFrame(
        [
            {"perturbagen": "pah_tobacco_like", "gene": "TP53", "score": 2.0},
            {"perturbagen": "pah_tobacco_like", "gene": "MDM2", "score": 2.0},
        ]
    ).to_csv(lincs_path, index=False)
    module_map_path = tmp_path / "module_map.csv"
    pd.DataFrame(
        [
            {"gene": "TP53", "module": "p53_checkpoint"},
            {"gene": "MDM2", "module": "p53_checkpoint"},
        ]
    ).to_csv(module_map_path, index=False)
    bundle = build_calibration_bundle(lincs_path=lincs_path, lincs_module_map=module_map_path)
    cfg = SimConfig(n=20, months=6, seed=3, archetype_prior={"pah_tobacco_like": 1.0})

    base, _ = simulate_cohort(cfg)
    calibrated, _ = simulate_cohort(cfg, calibration=bundle)

    assert float(calibrated["tx_p53_checkpoint"].mean()) > float(base["tx_p53_checkpoint"].mean())
