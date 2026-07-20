"""状态管理器模块。

提供实验状态的持久化存储与更新功能，
支持状态的保存、加载和基于工具结果的状态转换。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from hemzero.common.io import atomic_json
from hemzero.common.schemas import ExperimentState, Status, ToolResult


class StateManager:
    """实验状态管理器。

    负责实验状态的持久化存储，支持状态的保存、加载
    以及基于工具执行结果的状态转换。

    Attributes:
        state_path: 状态文件路径。
    """

    def __init__(self, state_path: Path):
        """初始化状态管理器。

        Args:
            state_path: 状态文件的存储路径。
        """
        self.state_path = state_path

    def save(self, state: ExperimentState) -> None:
        """保存实验状态到文件。

        Args:
            state: 要保存的实验状态对象。
        """
        atomic_json(self.state_path, asdict(state))

    def load(self) -> ExperimentState:
        """从文件加载实验状态。

        Returns:
            加载的实验状态对象。
        """
        return ExperimentState(**json.loads(self.state_path.read_text(encoding="utf-8")))

    def apply(self, state: ExperimentState, result: ToolResult) -> None:
        """将工具执行结果应用到实验状态。

        根据工具执行结果更新实验状态，包括当前阶段、
        已完成列表、阻塞字典和失败字典。

        Args:
            state: 要更新的实验状态对象。
            result: 工具执行结果。
        """
        state.current_stage = result.tool
        if result.status == Status.COMPLETE and result.tool not in state.completed:
            state.completed.append(result.tool)
        elif result.status == Status.BLOCKED:
            state.blocked[result.tool] = result.summary
        elif result.status == Status.FAILED:
            state.failed[result.tool] = result.error or result.summary
        self.save(state)
