"""命令行接口模块。

提供HemoZero实验流程的命令行入口，
支持实验启动、恢复和冒烟测试等操作。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from pi_agent.runner import PiRunner


def main() -> None:
    """命令行主入口函数。

    解析命令行参数并执行相应的实验操作，
    包括启动新实验、恢复运行和冒烟测试。
    """
    parser = argparse.ArgumentParser(prog="hemozero")
    parser.add_argument("command", choices=["start", "resume", "smoke"])
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--until")
    parser.add_argument("--workflow", type=Path, help="workflow YAML; defaults to configs/experiment.yaml")
    parser.add_argument("--finalize", action="store_true")
    arguments = parser.parse_args()
    runner = PiRunner(arguments.project_root, workflow_path=arguments.workflow)
    if arguments.command in ("start", "smoke"):
        state = runner.start(run_id=arguments.run_id)
        if arguments.command == "smoke":
            state = runner.resume(Path(state.run_dir), until=arguments.until or "create_cv_splits")
    else:
        if arguments.run_dir is None:
            parser.error("--run-dir is required for resume")
        state = runner.resume(arguments.run_dir, until=arguments.until, finalize=arguments.finalize)
    print(json.dumps(asdict(state), indent=2, ensure_ascii=False))
