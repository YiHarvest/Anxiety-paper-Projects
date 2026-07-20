from __future__ import annotations

import json

import pandas as pd

from hemzero.common.hashing import sha256_file
from hemzero.common.io import atomic_json
from hemzero.common.schemas import ArtifactRef, Status, ToolResult
from hemzero.knowledge_graph.patient_subgraph import build_patient_evidence_card
from hemzero.knowledge_graph.snapshot import read_jsonl

from ._shared import ToolContext, require_file


def run(arguments: dict, context: ToolContext) -> ToolResult:
    config = context.config("knowledge_graph.yaml")["snapshot"]
    root = context.run_dir / "knowledge_graph" / "patient_subgraphs"
    root.mkdir(parents=True, exist_ok=True)
    if not bool(config.get("patient_subgraphs_enabled", False)):
        status_path = root / "_DISABLED.json"
        atomic_json(status_path, {"enabled": False, "reason": "first implementation prioritizes interaction-method validation"})
        return ToolResult("kg_build_patient_subgraph", Status.COMPLETE, "Patient subgraphs intentionally disabled by configuration",
                          [ArtifactRef(str(status_path), sha256_file(status_path), "application/json")],
                          {"enabled": False, "llm_access": "read_only"})
    evaluation = context.run_dir / "evidence" / "tool_evaluation.json"
    final_complete = evaluation.exists() and json.loads(evaluation.read_text()).get("status") == "complete"
    predictions = pd.read_csv(require_file(context.run_dir / "predictions" / "hemozero_holdout.csv", "hemozero_holdout"))
    decisions = read_jsonl(require_file(context.run_dir / "knowledge_graph" / "interaction_decisions.jsonl", "interaction_decisions"))
    global_decisions = [row for row in decisions if row.get("repeat") is None]
    artifacts = []
    for prediction in predictions.to_dict(orient="records"):
        card = build_patient_evidence_card(prediction, global_decisions, final_evaluation_complete=final_complete)
        path = root / f"{prediction['patient_id']}.json"
        atomic_json(path, card)
        artifacts.append(ArtifactRef(str(path), sha256_file(path), "application/json"))
    return ToolResult("kg_build_patient_subgraph", Status.COMPLETE, "Read-only patient evidence cards created after final evaluation",
                      artifacts, {"cards": len(artifacts), "contains_holdout_labels": False, "llm_access": "read_only"})
