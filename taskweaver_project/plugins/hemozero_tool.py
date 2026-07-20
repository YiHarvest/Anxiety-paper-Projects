from __future__ import annotations

import json
from pathlib import Path

from taskweaver.plugin import Plugin, register_plugin

from pi_agent.evidence_reader import EvidenceReader, READ_ONLY_KNOWLEDGE_OPERATIONS


@register_plugin
class HemoZeroToolPlugin(Plugin):
    """Read-only bridge from TaskWeaver to audit-passed exported evidence."""

    def __call__(self, tool_name: str, arguments_json: str, run_dir: str) -> str:
        project_root = Path(__file__).resolve().parents[2]
        allowed_root = (project_root / "artifacts" / "runs").resolve()
        resolved_run = Path(run_dir).expanduser().resolve()
        if not resolved_run.is_relative_to(allowed_root):
            raise ValueError(f"run_dir must be inside {allowed_root}")
        arguments = json.loads(arguments_json or "{}")
        if not isinstance(arguments, dict):
            raise TypeError("arguments_json must decode to a JSON object")
        if tool_name not in READ_ONLY_KNOWLEDGE_OPERATIONS:
            raise PermissionError(f"TaskWeaver may execute only read-only evidence queries: {tool_name}")
        result = EvidenceReader(resolved_run).knowledge_query(tool_name, arguments)
        return json.dumps(result, ensure_ascii=False, default=str)
