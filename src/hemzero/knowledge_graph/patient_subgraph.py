"""Read-only patient evidence-card builder; disabled before final evaluation."""

from __future__ import annotations

from typing import Any


def build_patient_evidence_card(
    prediction: dict[str, Any],
    interaction_decisions: list[dict[str, Any]],
    *,
    final_evaluation_complete: bool,
) -> dict[str, Any]:
    if not final_evaluation_complete:
        raise PermissionError("Patient evidence subgraphs require completed final holdout evaluation")
    return {
        "patient_id": prediction["patient_id"],
        "prediction": {key: value for key, value in prediction.items() if key != "target"},
        "approved_interactions": [
            row for row in interaction_decisions
            if row.get("status") in {"retain_supported", "retain_discovery"}
        ],
        "claims_must_be_grounded": True,
        "llm_permissions": "read_only",
    }
