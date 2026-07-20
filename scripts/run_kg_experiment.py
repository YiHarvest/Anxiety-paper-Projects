#!/usr/bin/env python3
"""One-command HemoZero knowledge-constrained distillation experiment."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from hemzero.knowledge_graph.client import Neo4jSettings, capability
from hemzero.orchestration.preflight import model_preflight
from pi_agent.runner import PiRunner


def preflight(root: Path) -> dict:
    load_dotenv(root / ".env", override=False)
    checks = model_preflight(root)
    kg_config = PiRunner(root).project_root / "configs" / "knowledge_graph.yaml"
    from hemzero.common.io import load_yaml

    status = capability(load_yaml(kg_config))
    checks.update({"neo4j_driver_available": status.available, "neo4j_configured": status.configured})
    if status.available and status.configured:
        try:
            from hemzero.knowledge_graph.client import Neo4jRepository
            registry_version = str(load_yaml(root / "configs" / "evidence_registry.yaml")["registry_version"])
            with Neo4jRepository(
                Neo4jSettings.from_config(load_yaml(kg_config)), evidence_registry_version=registry_version
            ) as repository:
                checks["neo4j_version"] = repository.server_version()
                checks["neo4j_connectivity"] = True
        except Exception as exc:
            checks["neo4j_connectivity"] = False
            checks["neo4j_error"] = str(exc)
    else:
        checks["neo4j_connectivity"] = False
    kg_fatal = [name for name in ("neo4j_driver_available", "neo4j_configured", "neo4j_connectivity") if not checks[name]]
    checks["fatal"] = list(checks.get("fatal", [])) + kg_fatal
    checks["pass"] = not checks["fatal"]
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run knowledge-constrained HemoZero")
    parser.add_argument("--run-id")
    parser.add_argument("--resume")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    checks = preflight(root)
    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)
    if not checks["pass"]:
        raise SystemExit("Preflight failed; no KG experiment was started")
    if args.preflight_only:
        return
    runner = PiRunner(root, workflow_path="configs/workflows/hemozero_kg.yaml")
    if args.resume:
        run_dir = root / runner.experiment["artifact_root"] / args.resume
        if not run_dir.is_dir():
            raise SystemExit(f"Run not found: {run_dir}")
    else:
        run_id = args.run_id or f"hemozero-kg-{datetime.now():%Y%m%d-%H%M%S}"
        run_dir = Path(runner.start(run_id=run_id).run_dir)
    state = runner.resume(run_dir)
    if set(state.completed) == set(runner.workflow["stages"]) and not state.blocked and not state.failed:
        state = runner.resume(run_dir, finalize=True)
    print(state, flush=True)
    if state.blocked or state.failed or set(state.completed) != set(runner.workflow["stages"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
