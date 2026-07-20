import json

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from hemzero.distillation.knowledge_constrained_distillation import build_ablation_arms
from hemzero.graph.lineage import build_lineage_graph
from hemzero.knowledge_graph.evidence_loader import validate_evidence_registry
from hemzero.knowledge_graph.cohort_graph import build_fold_associations
from hemzero.knowledge_graph.interaction_policy import decide_interaction
from hemzero.knowledge_graph.lineage_sync import canonical_pair, export_lineage_records, feature_entity_id
from hemzero.knowledge_graph.models import InteractionAssessment, InteractionStatus
from hemzero.knowledge_graph.client import RUN_SCOPED_EXPORT_QUERIES
from hemzero.knowledge_graph.schema import CONSTRAINT_QUERIES
from hemzero.knowledge_graph.snapshot import export_snapshot, read_jsonl
from pi_agent.evidence_reader import EvidenceReader


def assessment(**overrides):
    values = {
        "direct_derivation": False, "shares_component": False, "selection_frequency": 0.8,
        "direction_consistency": 0.8, "has_supporting_evidence": False, "has_conflicting_evidence": False,
    }
    values.update(overrides)
    return InteractionAssessment(**values)


@pytest.mark.parametrize(("value", "expected"), [
    (assessment(direct_derivation=True), InteractionStatus.REJECT_MECHANICAL),
    (assessment(selection_frequency=0.2), InteractionStatus.REJECT_UNSTABLE),
    (assessment(selection_frequency=None), InteractionStatus.REJECT_INSUFFICIENT),
    (assessment(has_supporting_evidence=True), InteractionStatus.RETAIN_SUPPORTED),
    (assessment(), InteractionStatus.RETAIN_DISCOVERY),
    (assessment(has_supporting_evidence=True, has_conflicting_evidence=True), InteractionStatus.REVIEW_CONFLICT),
])
def test_six_policy_outcomes(value, expected):
    assert decide_interaction(value, min_frequency=0.6, min_direction_consistency=0.6) == expected


def test_networkx_lineage_is_exported_without_recomputation():
    raw = ["IL6", "IL10", "TNFalpha", "CRP", "ACTH", "CORT"]
    formulas = {
        "IL6/IL10": {"numerator": "IL6", "denominator": "IL10"},
        "TNFalpha/IL10": {"numerator": "TNFalpha", "denominator": "IL10"},
        "CRP/IL10": {"numerator": "CRP", "denominator": "IL10"},
        "CORT/ACTH": {"numerator": "CORT", "denominator": "ACTH"},
        "CORT/IL6": {"numerator": "CORT", "denominator": "IL6"},
        "CORT/CRP": {"numerator": "CORT", "denominator": "CRP"},
        "IL6/TNFalpha": {"numerator": "IL6", "denominator": "TNFalpha"},
        "CRP/IL6": {"numerator": "CRP", "denominator": "IL6"},
        "ACTH/IL6": {"numerator": "ACTH", "denominator": "IL6"},
    }
    graph = build_lineage_graph(raw, formulas)
    nodes, edges = export_lineage_records(graph)
    assert isinstance(graph, nx.MultiDiGraph)
    assert len(nodes) == 15
    assert len(edges) == graph.number_of_edges() == 78
    assert feature_entity_id("IL6/IL10") == "IL6__DIV__IL10"
    assert canonical_pair("CORT", "IL6") == "CORT::IL6"
    assert {row["relationship_type"] for row in edges} == {
        "NUMERATOR_OF", "DENOMINATOR_OF", "DERIVED_FROM", "SHARES_COMPONENT"
    }


def test_schema_constraints_are_idempotent_and_named():
    assert len(CONSTRAINT_QUERIES) >= 10
    assert all("CREATE CONSTRAINT" in query and "IF NOT EXISTS" in query for query in CONSTRAINT_QUERIES)
    assert "RETURN DISTINCT properties(n)" not in RUN_SCOPED_EXPORT_QUERIES["Model"]


def test_empty_manual_registry_is_valid_but_placeholder_is_not():
    assert validate_evidence_registry({"evidence": []}, {"IL6", "ANXIETY"}) == []
    with pytest.raises(ValueError, match="study requires"):
        validate_evidence_registry({"evidence": [{
            "assertion_id": "EV1", "subject": {"type": "Biomarker", "id": "IL6"},
            "predicate": "ASSOCIATED_WITH", "object": {"type": "Disease", "id": "ANXIETY"},
            "study": {"study_id": "PMID_x"}, "evidence_grade": "B", "accepted": True, "reviewer": "manual",
        }]}, {"IL6", "ANXIETY"})


def test_confirmatory_and_discovery_arms_are_separate():
    summaries = [
        {"summary_id": "s1", "pair_id": "A::B", "features": ["A", "B"], "frequency": 0.8, "direction_consistency": 0.9, "score": 1.0},
        {"summary_id": "s2", "pair_id": "A::C", "features": ["A", "C"], "frequency": 0.8, "direction_consistency": 0.9, "score": 0.5},
    ]
    decisions = [
        {"summary_id": "s1", "status": "retain_supported", "assessment": {"direct_derivation": False, "shares_component": False}},
        {"summary_id": "s2", "status": "retain_discovery", "assessment": {"direct_derivation": False, "shares_component": False}},
    ]
    arms = build_ablation_arms(summaries, decisions, min_frequency=0.6, min_direction_consistency=0.6, max_interactions=2)
    assert [row["pair_id"] for row in arms["M3-C"]] == ["A::B"]
    assert {row["pair_id"] for row in arms["M3-D"]} == {"A::B", "A::C"}


def test_evidence_reader_denies_writes_and_unfrozen_reads(tmp_path):
    run = tmp_path / "run"
    (run / "evidence").mkdir(parents=True)
    (run / "evidence" / "audit.json").write_text(json.dumps({"pass": True}), encoding="utf-8")
    reader = EvidenceReader(run)
    with pytest.raises(PermissionError, match="not allowed"):
        reader.knowledge_query("write_evidence", {})
    with pytest.raises(PermissionError, match="frozen"):
        reader.knowledge_query("get_interaction_decision", {"pair_id": "A::B"})


def test_cohort_associations_are_fold_local_and_noncausal():
    rng = np.random.default_rng(284)
    first = rng.normal(size=80)
    frame = pd.DataFrame({"A": first, "B": first + rng.normal(scale=0.1, size=80), "C": rng.normal(size=80)})
    rows = build_fold_associations(
        frame, run_id="run", repeat=0, fold=1, spearman_min_absolute=0.2,
        partial_min_absolute=0.1, graphical_lasso_alpha=0.05,
    )
    assert {row["method"] for row in rows} == {"spearman", "partial_correlation", "graphical_lasso"}
    assert all(row["source_partition"] == "outer_training_fold" and row["uses_target"] is False for row in rows)
    assert {row["relationship_type"] for row in rows} <= {"CORRELATED_WITH", "CONDITIONALLY_ASSOCIATED_WITH"}


def test_snapshot_scopes_static_entities_and_lineage_to_current_project(tmp_path):
    class Reader:
        def server_version(self):
            return "test"

        def export_nodes(self, label, *, run_id, allowed_ids=None):
            if allowed_ids is not None:
                return [{"id": value} for value in allowed_ids]
            return []

        def export_lineage_relationships(self, feature_ids):
            assert feature_ids == ["IL6", "IL6__DIV__IL10"]
            return [{"source": "IL6", "target": "IL6__DIV__IL10", "relationship_type": "NUMERATOR_OF"}]

        def export_run_relationships(self, run_id):
            assert run_id == "run-1"
            return [{"source": "run-1", "target": "ANXIETY", "relationship_type": "PREDICTS"}]

    project = tmp_path / "project"
    run = project / "artifacts" / "runs" / "run-1"
    project.mkdir()
    paths = [project / name for name in ("lineage.yaml", "evidence.yaml", "policy.yaml")]
    for path in paths:
        path.write_text("version: 1\n", encoding="utf-8")
    export_snapshot(
        Reader(), project_root=project, run_dir=run, evidence_registry_version="v1",
        lineage_config_path=paths[0], evidence_registry_path=paths[1], interaction_policy_path=paths[2],
        entity_ids={
            "Biomarker": ["IL6"], "DerivedRatio": ["IL6__DIV__IL10"],
            "BiologicalSystem": ["Inflammation"], "Disease": ["ANXIETY"],
        },
    )
    ids = {row["id"] for row in read_jsonl(run / "knowledge_graph" / "entities.jsonl")}
    assert ids == {"IL6", "IL6__DIV__IL10", "Inflammation", "ANXIETY"}
    assert read_jsonl(run / "knowledge_graph" / "run_relationships.jsonl") == [
        {"source": "run-1", "target": "ANXIETY", "relationship_type": "PREDICTS"}
    ]
