#!/usr/bin/env python3
from pathlib import Path

from pi_agent.runner import PiRunner


if __name__ == "__main__":
    runner = PiRunner(Path(__file__).resolve().parents[1])
    state = runner.start()
    print(runner.resume(Path(state.run_dir), until="create_cv_splits"))
