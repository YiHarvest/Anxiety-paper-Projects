"""ToolRegistry-compatible name for the idempotent KG schema initializer."""

from __future__ import annotations

from . import kg_initialize
from ._shared import ToolContext


def run(arguments: dict, context: ToolContext):
    result = kg_initialize.run(arguments, context)
    result.tool = "kg_initialize_schema"
    return result
