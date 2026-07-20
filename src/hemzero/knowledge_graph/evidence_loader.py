"""Strict loader for the small, manually reviewed external evidence registry."""

from __future__ import annotations

from typing import Any

from .schema import EVIDENCE_PREDICATES


ENTITY_TYPES = frozenset({"Biomarker", "DerivedRatio", "Disease"})
EVIDENCE_ROLES = frozenset({"supporting", "conflicting"})
GRADES = frozenset({"A", "B", "C", "D"})


def validate_evidence_registry(registry: dict[str, Any], known_entity_ids: set[str]) -> list[dict[str, Any]]:
    """Validate provenance fields; never infer or auto-accept evidence."""

    evidence = registry.get("evidence", [])
    if not isinstance(evidence, list):
        raise ValueError("evidence_registry.evidence must be a list")
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, row in enumerate(evidence):
        if not isinstance(row, dict):
            raise ValueError(f"evidence[{index}] must be a mapping")
        assertion_id = str(row.get("assertion_id", ""))
        if not assertion_id or assertion_id in seen:
            raise ValueError(f"Missing or duplicate assertion_id: {assertion_id!r}")
        seen.add(assertion_id)
        subject, object_ = row.get("subject", {}), row.get("object", {})
        for side, entity in (("subject", subject), ("object", object_)):
            if entity.get("type") not in ENTITY_TYPES:
                raise ValueError(f"{assertion_id}.{side}.type is not allowed")
            if entity.get("id") not in known_entity_ids:
                raise ValueError(f"{assertion_id}.{side}.id is unknown: {entity.get('id')}")
        if row.get("predicate") not in EVIDENCE_PREDICATES:
            raise ValueError(f"{assertion_id}.predicate is not allowed")
        if row.get("evidence_role", "supporting") not in EVIDENCE_ROLES:
            raise ValueError(f"{assertion_id}.evidence_role must be supporting or conflicting")
        if row.get("evidence_grade") not in GRADES:
            raise ValueError(f"{assertion_id}.evidence_grade must be one of {sorted(GRADES)}")
        study = row.get("study", {})
        if not study.get("study_id") or not study.get("title") or not study.get("study_design"):
            raise ValueError(f"{assertion_id}.study requires study_id, title, and study_design")
        if row.get("accepted") is not True:
            raise ValueError(f"{assertion_id} must be explicitly accepted by a reviewer")
        if row.get("reviewer") in (None, "", "automatic"):
            raise ValueError(f"{assertion_id} requires a named/manual reviewer")
        validated.append(row)
    return validated
