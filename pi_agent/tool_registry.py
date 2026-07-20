"""工具注册表模块。

提供工具的动态注册与执行功能，
通过模块名称映射实现工具的按需加载和统一接口调用。
"""

from __future__ import annotations

import importlib

from hemzero.common.schemas import ToolResult
from tools._shared import ToolContext


TOOL_MODULES = {
    name: f"tools.{name}" for name in (
        "inspect_dataset", "validate_ratios", "build_lineage_graph", "create_cv_splits", "train_baselines",
        "train_tabpfn_views", "hemzero_fusion", "tabdistill_ebm", "pysr_distillation", "evaluation", "audit", "reporting",
        "kg_preflight", "kg_initialize", "kg_initialize_schema", "kg_sync_lineage", "kg_import_evidence", "kg_build_fold_associations",
        "run_tabdistill", "aggregate_interaction_stability", "kg_assess_interactions", "train_kg_ebm_students",
        "kg_build_patient_subgraph", "kg_export_snapshot", "kg_audit",
    )
}


class ToolRegistry:
    """工具注册表。

    管理实验工具的注册信息，支持动态加载和执行。

    Attributes:
        names: 已注册工具名称列表。
    """

    @property
    def names(self) -> list[str]:
        """获取所有已注册工具名称。

        Returns:
            工具名称字符串列表。
        """
        return list(TOOL_MODULES)

    def execute(self, name: str, arguments: dict, context: ToolContext) -> ToolResult:
        """执行指定工具。

        根据工具名称动态加载模块并执行run函数。

        Args:
            name: 工具名称。
            arguments: 工具执行参数字典。
            context: 工具执行上下文，包含项目路径等信息。

        Returns:
            工具执行结果对象。

        Raises:
            KeyError: 工具名称未注册时抛出。
            TypeError: 工具返回类型不正确时抛出。
        """
        if name not in TOOL_MODULES:
            raise KeyError(f"Unknown tool: {name}")
        result = importlib.import_module(TOOL_MODULES[name]).run(arguments, context)
        if not isinstance(result, ToolResult):
            raise TypeError(f"Tool {name} returned {type(result).__name__}, expected ToolResult")
        return result
