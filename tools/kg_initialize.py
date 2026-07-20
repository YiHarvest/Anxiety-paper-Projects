from __future__ import annotations

from hemzero.common.hashing import sha256_file
from hemzero.common.schemas import ArtifactRef, Status, ToolResult
from hemzero.knowledge_graph.schema import SCHEMA_VERSION, schema_cypher

from ._kg_shared import open_repository
from ._shared import ToolContext


def run(arguments: dict, context: ToolContext) -> ToolResult:
    output = context.run_dir / "knowledge_graph" / "schema.cypher"
    output.write_text(schema_cypher(), encoding="utf-8")
    try:
        with open_repository(context) as repository:
            repository.initialize_schema()
            version = repository.server_version()
    except Exception as exc:
        return ToolResult("kg_initialize", Status.BLOCKED, f"Neo4j schema initialization failed: {exc}")
    return ToolResult("kg_initialize", Status.COMPLETE, "Idempotent Neo4j uniqueness constraints initialized",
                      [ArtifactRef(str(output), sha256_file(output), "text/plain")],
                      {"schema_version": SCHEMA_VERSION, "neo4j_version": version})
