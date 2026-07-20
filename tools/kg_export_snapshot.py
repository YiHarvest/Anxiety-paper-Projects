from __future__ import annotations

from hemzero.common.hashing import sha256_file
from hemzero.common.schemas import ArtifactRef, Status, ToolResult
from hemzero.knowledge_graph.lineage_sync import feature_entity_id
from hemzero.knowledge_graph.snapshot import export_snapshot

from ._kg_shared import open_repository
from ._shared import ToolContext


def run(arguments: dict, context: ToolContext) -> ToolResult:
    registry = context.config("evidence_registry.yaml")
    biological = context.config("biological_entities.yaml")
    entity_ids = {
        "Biomarker": [str(row["id"]) for row in biological["biomarkers"]],
        "DerivedRatio": [feature_entity_id(name) for name in context.config("ratio_formulas.yaml")],
        "BiologicalSystem": [str(row["id"]) for row in biological["systems"]],
        "Disease": [str(row["id"]) for row in biological["diseases"]],
    }
    try:
        with open_repository(context) as repository:
            result = export_snapshot(
                repository, project_root=context.project_root, run_dir=context.run_dir,
                evidence_registry_version=str(registry["registry_version"]),
                lineage_config_path=context.project_root / "configs" / "ratio_formulas.yaml",
                evidence_registry_path=context.project_root / "configs" / "evidence_registry.yaml",
                interaction_policy_path=context.project_root / "configs" / "interaction_policy.yaml",
                entity_ids=entity_ids,
            )
    except Exception as exc:
        return ToolResult("kg_export_snapshot", Status.BLOCKED, f"Knowledge-graph snapshot export failed: {exc}")
    paths = [path for path in sorted((context.run_dir / "knowledge_graph").iterdir()) if path.is_file()]
    return ToolResult("kg_export_snapshot", Status.COMPLETE, "Neo4j state exported as immutable run-scoped evidence files",
                      [ArtifactRef(str(path), sha256_file(path)) for path in paths],
                      {"files": len(paths), **result["statistics"]})
