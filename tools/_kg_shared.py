"""Shared Neo4j setup for deterministic KG tools (not exposed as a tool)."""

from __future__ import annotations

from dotenv import load_dotenv

from hemzero.knowledge_graph.client import Neo4jRepository, Neo4jSettings, capability

from ._shared import ToolContext


def open_repository(context: ToolContext) -> Neo4jRepository:
    load_dotenv(context.project_root / ".env", override=False)
    config = context.config("knowledge_graph.yaml")
    status = capability(config)
    if not status.available or not status.configured:
        raise RuntimeError(status.reason)
    registry = context.config("evidence_registry.yaml")
    registry_version = str(registry["registry_version"])
    return Neo4jRepository(
        Neo4jSettings.from_config(config), evidence_registry_version=registry_version,
        evidence_registry_empty=not bool(registry.get("evidence")),
    )
