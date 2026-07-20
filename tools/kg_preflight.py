from __future__ import annotations

from dotenv import load_dotenv

from hemzero.common.schemas import Status, ToolResult
from hemzero.knowledge_graph.client import Neo4jSettings, capability

from ._kg_shared import open_repository
from ._shared import ToolContext


def run(arguments: dict, context: ToolContext) -> ToolResult:
    load_dotenv(context.project_root / ".env", override=False)
    config = context.config("knowledge_graph.yaml")
    status = capability(config)
    if not status.available or not status.configured:
        return ToolResult("kg_preflight", Status.BLOCKED, status.reason, evidence={"neo4j": status.__dict__})
    try:
        with open_repository(context) as repository:
            version = repository.server_version()
    except Exception as exc:
        return ToolResult("kg_preflight", Status.BLOCKED, f"Neo4j connectivity failed: {exc}")
    return ToolResult("kg_preflight", Status.COMPLETE, "Neo4j driver, credentials and connectivity verified",
                      evidence={"neo4j_version": version, "database": Neo4jSettings.from_config(config).database})
