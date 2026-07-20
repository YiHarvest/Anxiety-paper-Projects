#!/usr/bin/env python3
"""One-command, resumable HemoZero primary experiment."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("JULIA_NUM_THREADS", "2")

from hemzero.orchestration.preflight import model_preflight
from pi_agent.runner import PiRunner


def preflight(root: Path) -> dict:
    return model_preflight(root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete HemoZero experiment")
    parser.add_argument("--run-id", help="new run id; generated automatically when omitted")
    parser.add_argument("--resume", help="resume an existing unfrozen run id")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    checks = preflight(root)
    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)
    if not checks["pass"]:
        raise SystemExit("Preflight failed; no experiment was started")
    if args.preflight_only:
        return
    runner = PiRunner(root)
    if args.resume:
        run_dir = root / runner.experiment["artifact_root"] / args.resume
        if not run_dir.is_dir():
            raise SystemExit(f"Run not found: {run_dir}")
    else:
        run_id = args.run_id or f"hemozero-full-origin-{datetime.now():%Y%m%d-%H%M%S}"
        state = runner.start(run_id=run_id)
        run_dir = Path(state.run_dir)
    state = runner.resume(run_dir)
    expected = set(runner.workflow["stages"])
    if set(state.completed) == expected and not state.blocked and not state.failed:
        state = runner.resume(run_dir, finalize=True)
    print(state, flush=True)
    if state.blocked or state.failed or set(state.completed) != expected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
