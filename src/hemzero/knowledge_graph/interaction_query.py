"""Build policy assessments from NetworkX facts, stability and Neo4j evidence."""

from __future__ import annotations

from typing import Protocol

import networkx as nx

from .lineage_sync import feature_entity_id
from .models import EvidenceMatch, InteractionAssessment


class EvidenceReader(Protocol):
    def direct_evidence(self, feature_a_id: str, feature_b_id: str) -> EvidenceMatch: ...


def lineage_facts(graph: nx.MultiDiGraph, feature_a: str, feature_b: str) -> tuple[bool, bool]:
    direct = False
    shared = False
    for source, target in ((feature_a, feature_b), (feature_b, feature_a)):
        relations = {data.get("relation") for data in graph.get_edge_data(source, target, default={}).values()}
        direct = direct or bool(relations & {"derived_from", "numerator_of", "denominator_of"})
        shared = shared or "shared_component" in relations
    return direct, shared


def assess_interaction(
    summary: dict,
    graph: nx.MultiDiGraph,
    evidence_reader: EvidenceReader,
) -> InteractionAssessment:
    feature_a, feature_b = summary["features"]
    direct, shared = lineage_facts(graph, feature_a, feature_b)
    evidence = evidence_reader.direct_evidence(feature_entity_id(feature_a), feature_entity_id(feature_b))
    return InteractionAssessment(
        direct_derivation=direct,
        shares_component=shared,
        selection_frequency=summary.get("frequency"),
        direction_consistency=summary.get("direction_consistency"),
        has_supporting_evidence=evidence.has_support,
        has_conflicting_evidence=evidence.has_conflict,
        supporting_assertion_ids=evidence.supporting_assertion_ids,
        conflicting_assertion_ids=evidence.conflicting_assertion_ids,
    )
