import networkx as nx

from hemzero.graph.interaction_filter import filter_interactions
from hemzero.graph.lineage import build_lineage_graph


def test_lineage_filter_rejects_raw_derived_pair():
    graph = build_lineage_graph(["A", "B", "C"], {"A/B": {"numerator": "A", "denominator": "B"}})
    accepted, rejected = filter_interactions(
        [{"features": ["A", "A/B"], "frequency": 0.9, "score": 2.0},
         {"features": ["A", "C"], "frequency": 0.8, "score": 1.0}],
        graph,
    )
    assert accepted[0]["features"] == ["A", "C"]
    assert rejected[0]["rejection_reason"] == "lineage_redundancy"


def test_lineage_is_a_networkx_graph():
    graph = build_lineage_graph(["A", "B"], {"A/B": {"numerator": "A", "denominator": "B"}})
    assert isinstance(graph, nx.MultiDiGraph)
    assert graph.nodes["A/B"]["kind"] == "derived"


def test_interaction_filter_rejects_direction_instability():
    graph = build_lineage_graph(["A", "B"], {})
    accepted, rejected = filter_interactions(
        [{"features": ["A", "B"], "frequency": 0.8, "direction_consistency": 0.2, "score": 1.0}],
        graph,
    )
    assert not accepted
    assert rejected[0]["rejection_reason"] == "unstable_direction"
