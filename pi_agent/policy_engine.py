"""策略引擎模块。

提供基于依赖关系的确定性阶段调度逻辑，
根据工作流定义自动计算下一个可执行阶段。
"""

from __future__ import annotations

from hemzero.knowledge_graph.interaction_policy import decide_interaction
from hemzero.knowledge_graph.models import InteractionAssessment, InteractionStatus


class PolicyEngine:
    """策略引擎，基于工作流定义进行确定性阶段调度。

    根据阶段依赖关系、已完成阶段、阻塞状态和失败状态，
    确定下一个可执行的实验阶段。

    Attributes:
        stages: 工作流阶段定义字典。
    """
    def __init__(self, workflow: dict):
        """初始化策略引擎。

        Args:
            workflow: 工作流配置字典，包含stages键定义各阶段。
        """
        self.stages = workflow["stages"]

    def next_stage(self, completed: list[str], blocked: dict[str, str], failed: dict[str, str]) -> str | None:
        """计算下一个可执行阶段。

        根据依赖关系、完成状态、阻塞和失败情况，
        确定下一个可执行的实验阶段。

        Args:
            completed: 已完成的阶段名称列表。
            blocked: 被阻塞的阶段字典，键为阶段名，值为阻塞原因。
            failed: 已失败的阶段字典，键为阶段名，值为失败信息。

        Returns:
            下一个可执行阶段名称，若无可用阶段则返回None。
        """
        if failed:
            return None
        completed_set = set(completed)
        for stage, definition in self.stages.items():
            if stage in completed_set or stage in blocked:
                continue
            if set(definition.get("requires", [])) <= completed_set:
                return stage
        return None

    def tool_for(self, stage: str) -> str:
        """获取指定阶段对应的工具名称。

        Args:
            stage: 阶段名称。

        Returns:
            该阶段对应的工具名称。
        """
        return self.stages[stage]["tool"]

    @staticmethod
    def decide_interaction(
        assessment: InteractionAssessment,
        policy_config: dict,
    ) -> InteractionStatus:
        """Make the final interaction decision from computed facts and KG evidence."""

        return decide_interaction(
            assessment,
            min_frequency=float(policy_config["min_frequency"]),
            min_direction_consistency=float(policy_config["min_direction_consistency"]),
        )
