"""Theory graph construction and export."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx

from ._branding import GRAPH_EDGES_JSON_FILENAME, GRAPH_GRAPHML_FILENAME

if TYPE_CHECKING:
    from .calibration.bundle import CalibrationBundle


def build_theory_graph(calibration: CalibrationBundle | None = None) -> nx.DiGraph:
    """Build the default directed ICg-CaST theory graph.

    If ``calibration`` provides ``graph_edges`` or ``graph_node_attributes``
    (e.g. from AOP-Wiki / AOP-DB exports) they are merged into the default
    theory graph. The base graph is otherwise unchanged.
    """
    graph = nx.DiGraph()
    nodes = {
        "chemical_exposure": "chemical",
        "KCC_vector": "KCC",
        "MIE": "MIE",
        "DNA_adducts": "KE",
        "ROS": "KE",
        "repair_failure": "KE",
        "epigenetic_memory": "KE",
        "transcriptomic_modules": "omics",
        "immune_surveillance": "KE",
        "proliferation": "KE",
        "mutational_signatures": "omics",
        "driver_acquisition": "ecology",
        "clonal_expansion": "ecology",
        "cancer_transition_risk": "AO",
    }
    for name, node_type in nodes.items():
        graph.add_node(name, node_type=node_type)

    graph.add_edges_from(
        [
            ("chemical_exposure", "KCC_vector"),
            ("KCC_vector", "MIE"),
            ("MIE", "DNA_adducts"),
            ("MIE", "ROS"),
            ("DNA_adducts", "repair_failure"),
            ("repair_failure", "mutational_signatures"),
            ("ROS", "epigenetic_memory"),
            ("KCC_vector", "transcriptomic_modules"),
            ("epigenetic_memory", "transcriptomic_modules"),
            ("transcriptomic_modules", "proliferation"),
            ("immune_surveillance", "clonal_expansion"),
            ("proliferation", "clonal_expansion"),
            ("mutational_signatures", "driver_acquisition"),
            ("driver_acquisition", "clonal_expansion"),
            ("clonal_expansion", "cancer_transition_risk"),
            ("epigenetic_memory", "cancer_transition_risk"),
        ]
    )

    if calibration is not None:
        if calibration.graph_edges:
            skipped = 0
            for row in calibration.graph_edges:
                src = str(row.get("source"))
                tgt = str(row.get("target"))
                if not src or not tgt or src == "None" or tgt == "None":
                    skipped += 1
                    continue
                attrs = {k: v for k, v in row.items() if k not in {"source", "target"}}
                attrs.setdefault("source_name", "AOP-Wiki")
                for node in (src, tgt):
                    if node not in graph:
                        graph.add_node(node, node_type="enriched", source_name="AOP-Wiki")
                graph.add_edge(src, tgt, **attrs)
            if skipped:
                warnings.warn(
                    f"skipped {skipped}/{len(calibration.graph_edges)} calibration "
                    "graph edges with missing or null source/target endpoints",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if calibration.graph_node_attributes:
            for node, attrs in calibration.graph_node_attributes.items():
                if node not in graph:
                    graph.add_node(node, node_type="enriched", source_name="AOP-DB")
                graph.nodes[node].update(dict(attrs))
                graph.nodes[node].setdefault("source_name", "AOP-DB")
        if calibration.provenance:
            graph.graph["provenance"] = dict(calibration.provenance)
    return graph


def _graphml_safe_copy(graph: nx.DiGraph) -> nx.DiGraph:
    """Return a copy of ``graph`` with dict/list attributes JSON-encoded.

    GraphML only supports scalar attribute types, so nested provenance dicts
    attached to the graph or to nodes/edges are serialised to JSON strings.
    """
    safe = graph.copy()

    def encode(attrs: dict[str, object]) -> None:
        for key, value in list(attrs.items()):
            if isinstance(value, dict | list | tuple | set):
                attrs[key] = json.dumps(value, default=str)

    encode(safe.graph)
    for _, attrs in safe.nodes(data=True):
        encode(attrs)
    for _, _, attrs in safe.edges(data=True):
        encode(attrs)
    return safe


def export_graph(graph: nx.DiGraph, outdir: str | Path, formats: tuple[str, ...] = ("graphml", "json")) -> dict[str, Path]:
    """Export graph files and return a mapping from format to written path."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    if "graphml" in formats:
        graphml = output_dir / GRAPH_GRAPHML_FILENAME
        nx.write_graphml(_graphml_safe_copy(graph), graphml)
        written["graphml"] = graphml
    if "json" in formats:
        json_path = output_dir / GRAPH_EDGES_JSON_FILENAME
        edges = [{"source": source, "target": target} for source, target in graph.edges()]
        json_path.write_text(json.dumps(edges, indent=2), encoding="utf-8")
        written["json"] = json_path
    return written


def write_theory_graph(
    outdir: str | Path,
    calibration: CalibrationBundle | None = None,
) -> dict[str, Path]:
    """Build and export the (optionally calibrated) theory graph."""
    return export_graph(build_theory_graph(calibration=calibration), outdir)
