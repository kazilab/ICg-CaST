from __future__ import annotations

import networkx as nx

from icg_cast import build_theory_graph, export_graph


def test_theory_graph_is_dag_and_exports(tmp_path) -> None:
    graph = build_theory_graph()
    written = export_graph(graph, tmp_path)

    assert graph.is_directed()
    assert nx.is_directed_acyclic_graph(graph)
    assert "cancer_transition_risk" in graph.nodes
    assert graph.nodes["cancer_transition_risk"]["node_type"] == "AO"
    assert written["graphml"].exists()
    assert written["json"].exists()
