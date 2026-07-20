from __future__ import annotations

from hemzero.common.hashing import sha256_file
from hemzero.common.schemas import ArtifactRef, Status, ToolResult
from hemzero.knowledge_graph.evidence_loader import validate_evidence_registry
from hemzero.knowledge_graph.lineage_sync import feature_entity_id
from hemzero.knowledge_graph.snapshot import write_jsonl

from ._kg_shared import open_repository
from ._shared import ToolContext


def run(arguments: dict, context: ToolContext) -> ToolResult:
    biological = context.config("biological_entities.yaml")
    registry = context.config("evidence_registry.yaml")
    formulas = context.config("ratio_formulas.yaml")
    biomarker_system = {row["id"]: row["system"] for row in biological["biomarkers"]}
    ratio_rows = []
    feature_system_rows = [
        {"feature_id": row["id"], "system_id": row["system"]} for row in biological["biomarkers"]
    ]
    for name, formula in formulas.items():
        numerator_system = biomarker_system[formula["numerator"]]
        denominator_system = biomarker_system[formula["denominator"]]
        system = numerator_system if numerator_system == denominator_system else "Cross_System"
        ratio_rows.append({"id": feature_entity_id(name), "properties": {"display_name": name, "system": system}})
        feature_system_rows.append({"feature_id": feature_entity_id(name), "system_id": system})
    known_ids = {row["id"] for key in ("biomarkers", "diseases") for row in biological[key]}
    known_ids |= {feature_entity_id(name) for name in formulas}
    try:
        assertions = validate_evidence_registry(registry, known_ids)
        assertions = [{**row, "registry_version": str(registry["registry_version"])} for row in assertions]
        with open_repository(context) as repository:
            repository.upsert_systems(biological["systems"])
            repository.upsert_diseases(biological["diseases"])
            repository.upsert_biomarkers([{"id": row["id"], "properties": row} for row in biological["biomarkers"]])
            repository.upsert_ratios(ratio_rows)
            repository.link_feature_systems(feature_system_rows)
            if assertions:
                repository.upsert_studies_and_assertions(assertions)
    except Exception as exc:
        return ToolResult("kg_import_evidence", Status.BLOCKED, f"Evidence import failed: {exc}")
    assertion_path = context.run_dir / "knowledge_graph" / "evidence_assertions.jsonl"
    study_path = context.run_dir / "knowledge_graph" / "studies.jsonl"
    write_jsonl(assertion_path, assertions)
    studies = list({row["study"]["study_id"]: row["study"] for row in assertions}.values())
    write_jsonl(study_path, studies)
    artifacts = [ArtifactRef(str(path), sha256_file(path), "application/x-ndjson") for path in (assertion_path, study_path)]
    return ToolResult("kg_import_evidence", Status.COMPLETE,
                      "Manually reviewed evidence registry imported without automatic literature extraction",
                      artifacts, {"assertions": len(assertions), "studies": len(studies),
                                  "registry_version": registry["registry_version"],
                                  "empty_registry": not assertions})
