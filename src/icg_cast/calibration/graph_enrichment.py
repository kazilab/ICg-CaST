"""qAOP graph enrichment from local AOP-Wiki / AOP-DB exports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import networkx as nx

from ..data_sources.common import DataSourceBundle


def enrich_theory_graph(
    base: nx.DiGraph,
    aopwiki: DataSourceBundle | None = None,
    aopdb: DataSourceBundle | None = None,
    node_alias_map: Mapping[str, str] | None = None,
    aopdb_node_column: str = "node_id",
) -> nx.DiGraph:
    """Return a copy of ``base`` enriched with AOP-Wiki edges and AOP-DB metadata.

    Args:
        base: default theory graph from :func:`icg_cast.graph.build_theory_graph`.
        aopwiki: optional AOP-Wiki adapter bundle with ``source`` / ``target``
            columns. Edges are added; missing nodes are created with
            ``source_name='AOP-Wiki'`` and ``node_type='enriched'``.
        aopdb: optional AOP-DB adapter bundle. Each row's columns are attached
            as node attributes for the node identified by ``aopdb_node_column``.
            Rows referring to unknown nodes are recorded as new ``enriched``
            nodes.
        node_alias_map: optional rename to harmonise external node names with
            the default theory-graph vocabulary (e.g. ``{"DNA adducts":
            "DNA_adducts"}``).

    Returns:
        A new ``nx.DiGraph`` with the enrichment applied. ``base`` is not
        mutated.
    """
    graph = base.copy()
    alias = dict(node_alias_map or {})

    def resolve(name: object) -> str:
        text = str(name)
        return alias.get(text, text)

    if aopwiki is not None:
        if aopwiki.metadata.get("adapter") != "aopwiki":
            raise ValueError("aopwiki must be a bundle from load_aopwiki_export")
        data = aopwiki.data
        missing = {"source", "target"} - set(data.columns)
        if missing:
            raise ValueError(f"AOP-Wiki bundle missing required columns: {sorted(missing)}")
        for row in data.to_dict(orient="records"):
            src = resolve(row["source"])
            tgt = resolve(row["target"])
            edge_attrs: dict[str, Any] = {
                k: v for k, v in row.items() if k not in {"source", "target"}
            }
            edge_attrs.setdefault("source_name", "AOP-Wiki")
            for node in (src, tgt):
                if node not in graph:
                    graph.add_node(node, node_type="enriched", source_name="AOP-Wiki")
            graph.add_edge(src, tgt, **edge_attrs)

    if aopdb is not None:
        if aopdb.metadata.get("adapter") != "aopdb":
            raise ValueError("aopdb must be a bundle from load_aopdb_export")
        data = aopdb.data
        if aopdb_node_column not in data.columns:
            raise ValueError(
                f"AOP-DB bundle missing node column {aopdb_node_column!r}; "
                f"available columns: {list(data.columns)}"
            )
        for row in data.to_dict(orient="records"):
            node = resolve(row[aopdb_node_column])
            attrs: dict[str, Any] = {
                k: v for k, v in row.items() if k != aopdb_node_column
            }
            if node not in graph:
                graph.add_node(node, node_type="enriched", source_name="AOP-DB")
            graph.nodes[node].update(attrs)
            graph.nodes[node].setdefault("source_name", "AOP-DB")

    provenance: dict[str, dict[str, Any]] = dict(graph.graph.get("provenance", {}))
    if aopwiki is not None:
        provenance["aopwiki"] = dict(aopwiki.provenance)
    if aopdb is not None:
        provenance["aopdb"] = dict(aopdb.provenance)
    if provenance:
        graph.graph["provenance"] = provenance
    return graph
