"""证据读取器模块。

提供实验运行目录中证据文件的安全读取功能，
支持路径安全验证和审计状态检查。
"""

from __future__ import annotations

import json
from pathlib import Path


READ_ONLY_KNOWLEDGE_OPERATIONS = frozenset({
    "get_patient_evidence_subgraph", "get_interaction_decision", "get_supporting_studies",
    "get_conflicting_studies", "get_prediction_provenance",
})


class EvidenceReader:
    """实验证据读取器。

    安全地读取实验运行目录中的证据文件，
    防止路径逃逸攻击。

    Attributes:
        run_dir: 实验运行目录路径。
    """

    def __init__(self, run_dir: Path):
        """初始化证据读取器。

        Args:
            run_dir: 实验运行目录路径。
        """
        self.run_dir = run_dir.resolve()

    def read_json(self, relative_path: str) -> dict:
        """安全读取JSON格式的证据文件。

        Args:
            relative_path: 相对于运行目录的文件路径。

        Returns:
            解析后的JSON数据字典。

        Raises:
            ValueError: 路径逃逸运行目录时抛出。
        """
        path = (self.run_dir / relative_path).resolve()
        if self.run_dir not in path.parents:
            raise ValueError("Evidence path escapes run directory")
        return json.loads(path.read_text(encoding="utf-8"))

    def audit_passed(self) -> bool:
        """检查审计是否通过。

        Returns:
            审计通过返回True，否则返回False。
        """
        path = self.run_dir / "evidence" / "audit.json"
        return path.exists() and self.read_json("evidence/audit.json").get("pass") is True

    def read_jsonl(self, relative_path: str) -> list[dict]:
        path = (self.run_dir / relative_path).resolve()
        if self.run_dir not in path.parents:
            raise ValueError("Evidence path escapes run directory")
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def knowledge_query(self, operation: str, arguments: dict) -> dict:
        """Execute one closed-set read over exported run evidence, never live graph writes."""

        if operation not in READ_ONLY_KNOWLEDGE_OPERATIONS:
            raise PermissionError(f"Read-only knowledge operation is not allowed: {operation}")
        if not (self.run_dir / "_FROZEN").exists():
            raise PermissionError("Knowledge queries require a frozen run snapshot")
        if not self.audit_passed():
            raise PermissionError("Knowledge queries require an audit-passed run")
        kg = "knowledge_graph"
        if operation == "get_patient_evidence_subgraph":
            return self.read_json(f"{kg}/patient_subgraphs/{str(arguments['patient_id'])}.json")
        if operation == "get_interaction_decision":
            pair_id = str(arguments["pair_id"])
            rows = self.read_jsonl(f"{kg}/interaction_decisions.jsonl")
            return {"decisions": [row for row in rows if row.get("pair_id") == pair_id]}
        if operation in {"get_supporting_studies", "get_conflicting_studies"}:
            assertion_ids = set(arguments.get("assertion_ids", []))
            role = "supporting" if operation == "get_supporting_studies" else "conflicting"
            assertions = [row for row in self.read_jsonl(f"{kg}/evidence_assertions.jsonl")
                          if row.get("assertion_id") in assertion_ids and row.get("evidence_role", "supporting") == role]
            study_ids = {row.get("study_id") for row in assertions}
            studies = [row for row in self.read_jsonl(f"{kg}/studies.jsonl") if row.get("study_id") in study_ids]
            return {"assertions": assertions, "studies": studies}
        patient_id = arguments.get("patient_id")
        patient = self.read_json(f"{kg}/patient_subgraphs/{patient_id}.json") if patient_id is not None else {}
        return {"run_id": self.run_dir.name, "patient": patient,
                "graph_manifest": self.read_json(f"{kg}/graph_manifest.json")}
