"""Orchestrator: assemble a CalibrationBundle from multiple local sources."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from ..data_sources import (
    load_aopdb_export,
    load_aopwiki_export,
    load_cosmic_sbs_matrix,
    load_lincs_signatures,
    load_toxcast_summary,
)
from .bundle import CalibrationBundle
from .kcc_priors import calibrate_kcc_priors_from_toxcast
from .modules import calibrate_transcript_modules_from_lincs
from .signatures import calibrate_signatures_from_cosmic


def build_calibration_bundle(
    *,
    cosmic_path: str | Path | None = None,
    cosmic_name_map: Mapping[str, str] | None = None,
    cosmic_signature_columns: list[str] | None = None,
    lincs_path: str | Path | None = None,
    lincs_metadata_path: str | Path | None = None,
    lincs_module_map: str | Path | pd.DataFrame | None = None,
    lincs_perturbagen_column: str = "perturbagen",
    lincs_gene_column: str = "gene",
    lincs_score_column: str = "score",
    toxcast_path: str | Path | None = None,
    toxcast_mapping: str | Path | pd.DataFrame | None = None,
    toxcast_chemical_column: str = "chemical_id",
    toxcast_assay_column: str = "assay",
    toxcast_hitcall_column: str = "hit_call",
    aopwiki_path: str | Path | None = None,
    aopdb_path: str | Path | None = None,
    aopdb_table: str | None = None,
) -> CalibrationBundle:
    """Assemble a :class:`CalibrationBundle` from any subset of local inputs.

    All inputs are optional. Each provided source contributes one slot of the
    bundle and an entry to ``bundle.provenance`` keyed by adapter name.
    """
    bundle = CalibrationBundle()

    if cosmic_path is not None:
        cosmic = load_cosmic_sbs_matrix(cosmic_path)
        labels, profiles = calibrate_signatures_from_cosmic(
            cosmic,
            name_map=cosmic_name_map,
            signature_columns=cosmic_signature_columns,
        )
        bundle.signature_labels = labels
        bundle.signature_profiles = {name: arr.tolist() for name, arr in profiles.items()}
        bundle.provenance["cosmic"] = dict(cosmic.provenance)

    if lincs_path is not None:
        if lincs_module_map is None:
            raise ValueError("lincs_module_map is required when lincs_path is provided")
        lincs = load_lincs_signatures(lincs_path, metadata_path=lincs_metadata_path)
        priors = calibrate_transcript_modules_from_lincs(
            lincs,
            lincs_module_map,
            perturbagen_column=lincs_perturbagen_column,
            gene_column=lincs_gene_column,
            score_column=lincs_score_column,
        )
        bundle.transcript_module_priors = priors.to_dict(orient="records")
        bundle.provenance["lincs"] = dict(lincs.provenance)

    if toxcast_path is not None:
        if toxcast_mapping is None:
            raise ValueError("toxcast_mapping is required when toxcast_path is provided")
        toxcast = load_toxcast_summary(toxcast_path)
        priors = calibrate_kcc_priors_from_toxcast(
            toxcast,
            toxcast_mapping,
            chemical_column=toxcast_chemical_column,
            assay_column=toxcast_assay_column,
            hitcall_column=toxcast_hitcall_column,
        )
        bundle.archetype_kcc = {name: list(vec) for name, vec in priors.items()}
        bundle.provenance["toxcast"] = dict(toxcast.provenance)

    if aopwiki_path is not None:
        aopwiki = load_aopwiki_export(aopwiki_path)
        bundle.graph_edges = aopwiki.data.to_dict(orient="records")
        bundle.provenance["aopwiki"] = dict(aopwiki.provenance)

    if aopdb_path is not None:
        aopdb = load_aopdb_export(aopdb_path, table=aopdb_table)
        node_col = None
        for candidate in ("node_id", "node", "id", "name"):
            if candidate in aopdb.data.columns:
                node_col = candidate
                break
        if node_col is None:
            raise ValueError(
                "AOP-DB export must contain one of: node_id, node, id, name"
            )
        attributes: dict[str, dict[str, object]] = {}
        for row in aopdb.data.to_dict(orient="records"):
            key = str(row[node_col])
            attrs = {k: v for k, v in row.items() if k != node_col}
            attributes.setdefault(key, {}).update(attrs)
        bundle.graph_node_attributes = attributes
        bundle.provenance["aopdb"] = dict(aopdb.provenance)

    return bundle
