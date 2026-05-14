"""Local AOP-Wiki export adapter."""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from .common import DataSourceBundle, make_bundle, read_local_table


def load_aopwiki_export(path: str | Path) -> DataSourceBundle:
    """Load a user-supplied AOP-Wiki CSV/TSV/JSON export."""
    data = read_local_table(path)
    metadata = {
        "adapter": "aopwiki",
        "required_columns_for_graph_mapping": ["source", "target"],
        "row_count": int(len(data)),
    }
    return make_bundle(path, "AOP-Wiki", data, metadata=metadata)


def map_aop_to_theory_graph(aop_export: DataSourceBundle) -> nx.DiGraph:
    """Map an edge-list-style AOP export onto a NetworkX directed graph."""
    data = aop_export.data
    missing = {"source", "target"} - set(data.columns)
    if missing:
        raise ValueError(f"AOP-Wiki graph mapping requires columns: {sorted(missing)}")
    graph = nx.DiGraph()
    for row in data.to_dict(orient="records"):
        source = str(row["source"])
        target = str(row["target"])
        edge_attrs = {k: v for k, v in row.items() if k not in {"source", "target"}}
        graph.add_node(source, source_name="AOP-Wiki")
        graph.add_node(target, source_name="AOP-Wiki")
        graph.add_edge(source, target, **edge_attrs)
    graph.graph["provenance"] = aop_export.provenance
    return graph
