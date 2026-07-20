from __future__ import annotations

import json

from hemzero.common.hashing import sha256_file
from hemzero.common.io import atomic_json
from hemzero.common.schemas import ArtifactRef, Status, ToolResult
from hemzero.knowledge_graph.audit import audit_snapshot
from hemzero.knowledge_graph.snapshot import read_jsonl

from ._shared import ToolContext


def _tool_evidence(context: ToolContext, name: str) -> dict:
    path = context.run_dir / "evidence" / f"tool_{name}.json"
    return json.loads(path.read_text()).get("evidence", {}) if path.exists() else {}


def run(arguments: dict, context: ToolContext) -> ToolResult:
    root = context.run_dir / "knowledge_graph"
    files = {
        "entities": "entities.jsonl", "evidence_assertions": "evidence_assertions.jsonl",
        "interaction_candidates": "interaction_candidates.jsonl", "interaction_summaries": "interaction_summaries.jsonl",
        "interaction_decisions": "interaction_decisions.jsonl", "lineage_relationships": "lineage_relationships.jsonl",
        "run_relationships": "run_relationships.jsonl", "fold_associations": "fold_associations.jsonl",
    }
    missing = [filename for filename in files.values() if not (root / filename).exists()]
    if missing:
        return ToolResult("kg_audit", Status.FAILED, "Required KG snapshot files are missing", error=str(missing))
    snapshot = {key: read_jsonl(root / filename) for key, filename in files.items()}
    graph_manifest_path = root / "graph_manifest.json"
    if not graph_manifest_path.exists():
        return ToolResult("kg_audit", Status.FAILED, "graph_manifest.json is missing")
    graph_manifest = json.loads(graph_manifest_path.read_text(encoding="utf-8"))
    manifest_errors = []
    for item in graph_manifest.get("files", []):
        path = (root / item["path"]).resolve()
        if root.resolve() not in path.parents or not path.exists():
            manifest_errors.append(f"missing_or_unsafe:{item['path']}")
        elif sha256_file(path) != item["sha256"]:
            manifest_errors.append(f"hash_mismatch:{item['path']}")
    findings = audit_snapshot(snapshot, known_run_id=context.run_dir.name)
    entity_types = [row.get("node_type") or row.get("measurement_type") for row in snapshot["entities"]]
    counts = {"biomarkers": sum(value in {"Biomarker", "raw"} for value in entity_types),
              "ratios": sum(value in {"DerivedRatio", "derived"} for value in entity_types)}
    tabdistill = _tool_evidence(context, "run_tabdistill")
    run_relationships = snapshot["run_relationships"]
    summary_ids = {str(row.get("summary_id")) for row in snapshot["interaction_summaries"]}
    left_summary_ids = {
        str(row.get("source")) for row in run_relationships if row.get("relationship_type") == "LEFT_FEATURE"
    }
    right_summary_ids = {
        str(row.get("source")) for row in run_relationships if row.get("relationship_type") == "RIGHT_FEATURE"
    }
    prediction_edges = [
        row for row in run_relationships
        if row.get("relationship_type") == "PREDICTS" and row.get("source") == context.run_dir.name
    ]
    extra_checks = {
        "required_snapshot_files": not missing,
        "graph_manifest_hashes_valid": not manifest_errors,
        "lineage_config_hash_valid": graph_manifest.get("lineage_config_hash") == sha256_file(context.project_root / "configs" / "ratio_formulas.yaml"),
        "evidence_registry_hash_valid": graph_manifest.get("evidence_registry_hash") == sha256_file(context.project_root / "configs" / "evidence_registry.yaml"),
        "interaction_policy_hash_valid": graph_manifest.get("interaction_policy_hash") == sha256_file(context.project_root / "configs" / "interaction_policy.yaml"),
        "six_biomarkers_present": counts["biomarkers"] == 6,
        "nine_ratios_present": counts["ratios"] == 9,
        "networkx_lineage_edges_preserved": len(snapshot["lineage_relationships"]) == 78,
        "all_candidates_persisted": len(snapshot["interaction_candidates"]) == int(tabdistill.get("candidates", -1)),
        "all_summaries_decided": len(snapshot["interaction_summaries"]) == len(snapshot["interaction_decisions"]),
        "all_summaries_link_left_feature": left_summary_ids == summary_ids,
        "all_summaries_link_right_feature": right_summary_ids == summary_ids,
        "run_predicts_one_disease": len(prediction_edges) == 1,
        "neo4j_not_policy_authority": _tool_evidence(context, "kg_assess_interactions").get("neo4j_made_decisions") is False,
        "holdout_labels_isolated": _tool_evidence(context, "evaluation").get("test_labels_used_for_tuning") is False,
        "fold_associations_run_scoped": all(
            row.get("run_id") == context.run_dir.name for row in snapshot["fold_associations"]
        ),
        "fold_associations_are_feature_only": all(
            row.get("source_partition") == "outer_training_fold" and row.get("uses_target") is False
            for row in snapshot["fold_associations"]
        ),
    }
    checks = {finding.check: finding.passed for finding in findings} | extra_checks
    tabpfn = _tool_evidence(context, "train_tabpfn_views")
    report = {
        "checks": checks, "findings": [finding.__dict__ for finding in findings], "missing": missing,
        "graph_manifest_errors": manifest_errors,
        "pass": all(checks.values()) and not missing,
        "scope_limitations": {
            "designed_outer_repeats": tabpfn.get("outer_repeats_designed"),
            "executed_outer_repeats": tabpfn.get("outer_repeats_executed"),
            "execution_profile": tabpfn.get("execution_profile"),
            "tabdistill_search_repeats": tabdistill.get("search_repeats_executed"),
            "tabdistill_folds_per_repeat": tabdistill.get("search_folds_per_repeat"),
        },
    }
    kg_path = root / "kg_audit.json"
    evidence_path = context.run_dir / "evidence" / "audit.json"
    atomic_json(kg_path, report)
    atomic_json(evidence_path, report)
    status = Status.COMPLETE if report["pass"] else Status.FAILED
    return ToolResult("kg_audit", status, "Knowledge graph, policy authority and holdout isolation audited",
                      [ArtifactRef(str(path), sha256_file(path), "application/json") for path in (kg_path, evidence_path)],
                      report, None if report["pass"] else "Knowledge-graph audit failed")
