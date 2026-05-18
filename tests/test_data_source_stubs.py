from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from icg_cast.data_sources import (
    calibration_provenance_payload,
    load_aopdb_export,
    load_aopwiki_export,
    load_cosmic_sbs_matrix,
    load_ctd_chemical_gene_disease,
    load_gdc_manifest,
    load_lincs_signatures,
    load_sigprofiler_activities,
    load_toxcast_summary,
    map_aop_to_theory_graph,
    validate_calibration_provenance,
    validate_provenance_record,
)
from icg_cast.signatures import mutation_context_labels


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_aopwiki_loader_maps_local_edge_list_to_graph(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "aopwiki_edges.csv",
        [
            {"source": "DNA_adducts", "target": "mutation_rate", "relationship": "increases"},
            {"source": "mutation_rate", "target": "driver_acquisition", "relationship": "increases"},
        ],
    )

    bundle = load_aopwiki_export(path)
    graph = map_aop_to_theory_graph(bundle)

    assert bundle.metadata["adapter"] == "aopwiki"
    assert bundle.provenance["source_name"] == "AOP-Wiki"
    assert len(bundle.provenance["sha256"]) == 64
    assert graph.has_edge("DNA_adducts", "mutation_rate")
    assert graph.nodes["DNA_adducts"]["source_name"] == "AOP-Wiki"


def test_cosmic_loader_validates_96_channel_local_matrix(tmp_path: Path) -> None:
    labels = mutation_context_labels()
    path = tmp_path / "cosmic_sbs.csv"
    pd.DataFrame(
        {
            "context": labels,
            "SBS4": [1.0 / len(labels)] * len(labels),
            "SBS24": [2.0 / len(labels)] * len(labels),
        }
    ).to_csv(path, index=False)

    bundle = load_cosmic_sbs_matrix(path)

    assert bundle.metadata["adapter"] == "cosmic"
    assert bundle.metadata["n_contexts"] == 96
    assert bundle.metadata["n_signatures"] == 2
    assert bundle.data.shape == (96, 3)


def test_cosmic_loader_rejects_non_96_channel_matrix(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "bad_cosmic.csv",
        [
            {"context": "A[C>A]A", "SBS4": 1.0},
            {"context": "A[C>A]C", "SBS4": 1.0},
        ],
    )

    with pytest.raises(ValueError, match="96 contexts"):
        load_cosmic_sbs_matrix(path)


def test_toxcast_loader_accepts_optional_mapping_file(tmp_path: Path) -> None:
    summary = _write_csv(
        tmp_path / "toxcast_summary.csv",
        [{"chemical_id": "DTXSID001", "assay": "DNA_damage", "hit_call": 1}],
    )
    mapping = _write_csv(
        tmp_path / "kcc_mapping.csv",
        [{"assay": "DNA_damage", "kcc_id": "KCC2", "evidence": "toy fixture"}],
    )

    bundle = load_toxcast_summary(summary, mapping_path=mapping)

    assert bundle.metadata["adapter"] == "toxcast"
    assert bundle.metadata["mapping_rows"] == 1
    assert bundle.data["chemical_id"].iloc[0] == "DTXSID001"


def test_other_adapters_load_local_mock_tables(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "mock.csv", [{"id": "row1", "value": 1.5}])

    loaders = [
        load_ctd_chemical_gene_disease,
        load_gdc_manifest,
        load_lincs_signatures,
        load_sigprofiler_activities,
    ]
    for loader in loaders:
        bundle = loader(path)
        assert len(bundle.data) == 1
        assert len(bundle.provenance["sha256"]) == 64


def test_aopdb_loader_can_read_local_sqlite_export(tmp_path: Path) -> None:
    path = tmp_path / "aopdb.sqlite"
    with sqlite3.connect(path) as con:
        con.execute("create table edges (source text, target text)")
        con.execute("insert into edges values ('MIE', 'KE')")

    bundle = load_aopdb_export(path, table="edges")

    assert bundle.metadata["adapter"] == "aopdb"
    assert bundle.data.to_dict(orient="records") == [{"source": "MIE", "target": "KE"}]


def test_adapters_reject_remote_paths() -> None:
    with pytest.raises(ValueError, match="local files only"):
        load_gdc_manifest("https://example.org/gdc_manifest.csv")


def test_provenance_schema_validation_rejects_incomplete_records(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "mock.csv", [{"id": "row1", "value": 1.5}])
    bundle = load_lincs_signatures(path)

    assert validate_provenance_record(bundle.provenance)["source_name"].startswith("LINCS L1000")

    broken = dict(bundle.provenance)
    broken["sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="sha256"):
        validate_provenance_record(broken)


def test_calibration_provenance_payload_is_versioned(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "mock.csv", [{"id": "row1", "value": 1.5}])
    bundle = load_lincs_signatures(path)

    payload = calibration_provenance_payload(
        {"lincs": bundle.provenance},
        coefficient_updates={"n_updates": 1},
    )

    assert payload["schema_version"] == "0.1"
    assert payload["lincs"]["sha256"] == bundle.provenance["sha256"]
    assert payload["coefficient_updates"]["n_updates"] == 1
    assert validate_calibration_provenance(payload) == payload
