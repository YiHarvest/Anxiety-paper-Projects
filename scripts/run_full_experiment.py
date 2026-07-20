#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from pi_agent.runner import PiRunner


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    args = parser.parse_args()
    runner = PiRunner(Path(__file__).resolve().parents[1])
    state = runner.start(run_id=args.run_id)
    state = runner.resume(Path(state.run_dir))
    expected = set(runner.workflow["stages"])
    if set(state.completed) == expected and not state.blocked and not state.failed:
        state = runner.resume(Path(state.run_dir), finalize=True)
    print(state)
