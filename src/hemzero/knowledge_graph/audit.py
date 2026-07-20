"""Pure audit rules for exported KG snapshots."""

from __future__ import annotations

from typing import Any

from .models import AuditFinding, InteractionStatus


def audit_snapshot(snapshot: dict[str, list[dict[str, Any]]], *, known_run_id: str) -> list[AuditFinding]:
    candidates = snapshot.get("interaction_candidates", [])
    decisions = snapshot.get("interaction_decisions", [])
    assertions = snapshot.get("evidence_assertions", [])
    candidate_ids = [str(row.get("candidate_id")) for row in candidates]
    decision_ids = [str(row.get("decision_id")) for row in decisions]
    allowed_statuses = {status.value for status in InteractionStatus}
    return [
        AuditFinding("candidate_ids_unique", len(candidate_ids) == len(set(candidate_ids)),
                     "Every TabDistill candidate must have one stable ID"),
        AuditFinding("decision_ids_unique", len(decision_ids) == len(set(decision_ids)),
                     "Every policy decision must have one stable ID"),
        AuditFinding("candidate_run_scope", all(row.get("run_id") == known_run_id for row in candidates),
                     "Candidates may not cross run boundaries"),
        AuditFinding("decision_vocabulary", all(row.get("status") in allowed_statuses for row in decisions),
                     "Decisions use the closed six-status vocabulary"),
        AuditFinding("manual_evidence_only", all(row.get("accepted") is True and row.get("reviewer") not in (None, "", "automatic") for row in assertions),
                     "Every imported assertion was explicitly accepted by a manual reviewer"),
        AuditFinding("no_holdout_label_edges", all(row.get("source_partition") != "independent_holdout" or row.get("uses_target") is False for rows in snapshot.values() for row in rows),
                     "No pre-evaluation KG record uses independent holdout labels"),
    ]
