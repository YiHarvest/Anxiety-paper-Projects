"""Deterministic file snapshot of Neo4j state for immutable run evidence."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from hemzero.common.hashing import sha256_file
from hemzero.common.io import atomic_json

from .schema import SCHEMA_VERSION, schema_cypher


class SnapshotReader(Protocol):
    def server_version(self) -> str: ...
    def export_nodes(
        self,
        label: str,
        *,
        run_id: str,
        allowed_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]: ...
    def export_lineage_relationships(self, feature_ids: list[str]) -> list[dict[str, Any]]: ...
    def export_run_relationships(self, run_id: str) -> list[dict[str, Any]]: ...


NODE_EXPORTS = {
    "entities.jsonl": ("Biomarker", "DerivedRatio", "BiologicalSystem", "Disease"),
    "studies.jsonl": ("Study",),
    "evidence_assertions.jsonl": ("EvidenceAssertion",),
    "interaction_candidates.jsonl": ("InteractionCandidate",),
    "interaction_summaries.jsonl": ("InteractionSummary",),
    "interaction_decisions.jsonl": ("InteractionDecision",),
    "model_runs.jsonl": ("ModelRun",),
    "cv_folds.jsonl": ("CVFold",),
    "models.jsonl": ("Model",),
}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _git_commit(project_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root, capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def export_snapshot(
    reader: SnapshotReader,
    *,
    project_root: Path,
    run_dir: Path,
    evidence_registry_version: str,
    lineage_config_path: Path,
    evidence_registry_path: Path,
    interaction_policy_path: Path,
    entity_ids: dict[str, list[str]],
) -> dict[str, Any]:
    root = run_dir / "knowledge_graph"
    root.mkdir(parents=True, exist_ok=True)
    (root / "schema.cypher").write_text(schema_cypher(), encoding="utf-8")
    snapshot: dict[str, list[dict[str, Any]]] = {}
    for filename, labels in NODE_EXPORTS.items():
        rows = [
            row
            for label in labels
            for row in reader.export_nodes(
                label,
                run_id=run_dir.name,
                allowed_ids=entity_ids.get(label),
            )
        ]
        rows.sort(key=lambda row: json.dumps(row, sort_keys=True, default=str))
        write_jsonl(root / filename, rows)
        snapshot[filename.removesuffix(".jsonl")] = rows
    feature_ids = entity_ids["Biomarker"] + entity_ids["DerivedRatio"]
    lineage = reader.export_lineage_relationships(feature_ids)
    write_jsonl(root / "lineage_relationships.jsonl", lineage)
    run_relationships = reader.export_run_relationships(run_dir.name)
    write_jsonl(root / "run_relationships.jsonl", run_relationships)
    decisions = snapshot["interaction_decisions"]
    write_jsonl(root / "rejected_interactions.jsonl", [r for r in decisions if str(r.get("status", "")).startswith("reject_")])
    write_jsonl(root / "conflicted_interactions.jsonl", [r for r in decisions if r.get("status") == "review_conflict"])
    statistics = {
        "snapshot_record_counts": {key: len(rows) for key, rows in snapshot.items()},
        "lineage_relationships": len(lineage),
        "run_relationships": len(run_relationships),
    }
    atomic_json(root / "kg_statistics.json", statistics)
    manifest = {
        "neo4j_version": reader.server_version(),
        "schema_version": SCHEMA_VERSION,
        "evidence_registry_version": evidence_registry_version,
        "lineage_config_hash": sha256_file(lineage_config_path),
        "evidence_registry_hash": sha256_file(evidence_registry_path),
        "interaction_policy_hash": sha256_file(interaction_policy_path),
        "git_commit": _git_commit(project_root),
        "run_id": run_dir.name,
        "exported_at": datetime.now(UTC).isoformat(),
    }
    manifest["files"] = [
        {"path": str(path.relative_to(root)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"graph_manifest.json", "kg_audit.json"}
    ]
    atomic_json(root / "graph_manifest.json", manifest)
    return {"manifest": manifest, "statistics": statistics, "snapshot": snapshot}
