"""Pi运行器模块。

提供实验流程的主编排功能，
协调策略引擎、状态管理和工具注册表，
实现完整的实验生命周期管理。
"""

from __future__ import annotations

from pathlib import Path

from hemzero.common.io import atomic_json, load_yaml
from hemzero.common.schemas import ExperimentState, Status
from hemzero.reporting.artifacts import ArtifactStore
from hemzero.tracking import tracked_run
from tools._shared import ToolContext

from .policy_engine import PolicyEngine
from .state_manager import StateManager
from .tool_registry import ToolRegistry


class PiRunner:
    """Pi风格实验运行器。

    编排完整的实验流程，包括实验初始化、
    状态恢复、阶段调度和结果持久化。

    Attributes:
        project_root: 项目根目录路径。
        experiment: 实验配置字典。
        workflow: 工作流配置字典。
        registry: 工具注册表实例。
    """

    def __init__(self, project_root: Path, *, workflow_path: Path | str | None = None):
        """初始化Pi运行器。

        Args:
            project_root: 项目根目录路径。
        """
        self.project_root = project_root.resolve()
        self.experiment = load_yaml(self.project_root / "configs" / "experiment.yaml")
        configured_workflow = workflow_path or self.experiment["workflow"]
        configured_workflow = Path(configured_workflow)
        self.workflow_path = configured_workflow if configured_workflow.is_absolute() else self.project_root / configured_workflow
        self.workflow = load_yaml(self.workflow_path)
        self.registry = ToolRegistry()

    def start(self, *, run_id: str | None = None) -> ExperimentState:
        """启动新实验运行。

        创建运行目录并初始化实验状态。

        Args:
            run_id: 可选的运行ID，未提供时自动生成。

        Returns:
            初始化后的实验状态对象。
        """
        run_dir = ArtifactStore(self.project_root / self.experiment["artifact_root"]).create_run(run_id)
        state = ExperimentState(run_id=run_dir.name, run_dir=str(run_dir))
        StateManager(run_dir / "evidence" / "state.json").save(state)
        return state

    def resume(self, run_dir: Path, *, until: str | None = None, finalize: bool = False) -> ExperimentState:
        """恢复并继续实验运行。

        从已有运行目录恢复状态，继续执行未完成的阶段。

        Args:
            run_dir: 运行目录路径。
            until: 可选的停止阶段名称，执行到此阶段后停止。
            finalize: 是否在完成后执行最终化操作。

        Returns:
            更新后的实验状态对象。
        """
        run_dir = run_dir.resolve()
        ArtifactStore.assert_writable(run_dir)
        manager = StateManager(run_dir / "evidence" / "state.json")
        state, policy = manager.load(), PolicyEngine(self.workflow)
        context = ToolContext(self.project_root, run_dir)
        with tracked_run(run_id=state.run_id, tracking_uri=self.project_root / self.experiment["tracking_uri"], parameters=self.experiment):
            while True:
                stage = policy.next_stage(state.completed, state.blocked, state.failed)
                if stage is None:
                    break
                result = self.registry.execute(policy.tool_for(stage), {}, context)
                atomic_json(run_dir / "evidence" / f"tool_{stage}.json", result.to_dict())
                manager.apply(state, result)
                if result.status in (Status.BLOCKED, Status.FAILED) or stage == until:
                    break
        if finalize:
            store = ArtifactStore(self.project_root / self.experiment["artifact_root"])
            state.artifact_manifest = str((run_dir / "manifest.json").resolve())
            manager.save(state)
            store.finalize(run_dir)
        return state
