from __future__ import annotations

from hemzero.common.hashing import sha256_file
from hemzero.common.schemas import ArtifactRef, Status, ToolResult
from hemzero.graph.lineage import build_lineage_graph
from hemzero.knowledge_graph.lineage_sync import export_lineage_records
from hemzero.knowledge_graph.snapshot import write_jsonl

from ._kg_shared import open_repository
from ._shared import ToolContext


def run(arguments: dict, context: ToolContext) -> ToolResult:
    dataset = context.config("dataset.yaml")
    graph = build_lineage_graph(dataset["raw_features"], context.config("ratio_formulas.yaml"))
    nodes, relationships = export_lineage_records(graph)
    biomarkers = [row for row in nodes if row["node_type"] == "Biomarker"]
    ratios = [row for row in nodes if row["node_type"] == "DerivedRatio"]
    try:
        with open_repository(context) as repository:
            repository.upsert_biomarkers(biomarkers)
            repository.upsert_ratios(ratios)
            repository.sync_lineage_relationships(relationships)
    except Exception as exc:
        return ToolResult("kg_sync_lineage", Status.BLOCKED, f"NetworkX lineage sync failed: {exc}")
    entity_path = context.run_dir / "knowledge_graph" / "entities.jsonl"
    relationship_path = context.run_dir / "knowledge_graph" / "lineage_relationships.jsonl"
    write_jsonl(entity_path, nodes)
    write_jsonl(relationship_path, relationships)
    artifacts = [ArtifactRef(str(path), sha256_file(path), "application/x-ndjson") for path in (entity_path, relationship_path)]
    return ToolResult("kg_sync_lineage", Status.COMPLETE, "Existing NetworkX lineage losslessly synchronized with Neo4j MERGE",
                      artifacts, {"nodes": len(nodes), "relationships": len(relationships), "networkx_retained": True})
