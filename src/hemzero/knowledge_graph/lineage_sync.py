"""Lossless adapter from the existing NetworkX lineage graph to KG records."""

from __future__ import annotations

from typing import Any

import networkx as nx


RELATIONSHIP_NAMES = {
    "numerator_of": "NUMERATOR_OF",
    "denominator_of": "DENOMINATOR_OF",
    "derived_from": "DERIVED_FROM",
    "shared_component": "SHARES_COMPONENT",
}


def feature_entity_id(feature: str) -> str:
    """Create a path-safe stable ID while keeping display names unchanged."""

    return str(feature).replace("/", "__DIV__")


def feature_display_name(entity_id: str) -> str:
    return str(entity_id).replace("__DIV__", "/")


def canonical_pair(feature_a: str, feature_b: str) -> str:
    left, right = sorted((feature_entity_id(feature_a), feature_entity_id(feature_b)))
    return f"{left}::{right}"


def export_lineage_records(
    graph: nx.MultiDiGraph,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Export every node and every keyed edge without recomputing lineage."""

    nodes: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    for node_id, attrs in graph.nodes(data=True):
        kind = str(attrs.get("kind", attrs.get("node_type", "unknown")))
        entity_id = feature_entity_id(str(node_id))
        node_type = "Biomarker" if kind == "raw" else "DerivedRatio" if kind == "derived" else "Unknown"
        properties = dict(attrs)
        properties.update({"id": entity_id, "display_name": str(node_id), "measurement_type": kind})
        if node_type == "DerivedRatio":
            properties.setdefault("deterministic", True)
        nodes.append({"id": entity_id, "node_type": node_type, "properties": properties})

    for source, target, edge_key, attrs in graph.edges(keys=True, data=True):
        relation = str(attrs.get("relation", edge_key))
        relationship_type = RELATIONSHIP_NAMES.get(relation)
        if relationship_type is None:
            raise ValueError(f"Unknown NetworkX lineage relation: {relation}")
        source_id, target_id = feature_entity_id(str(source)), feature_entity_id(str(target))
        relationships.append({
            "edge_id": f"{source_id}::{target_id}::{edge_key}",
            "source": source_id,
            "target": target_id,
            "relationship_type": relationship_type,
            "properties": {**dict(attrs), "networkx_edge_key": str(edge_key)},
        })
    return nodes, relationships
