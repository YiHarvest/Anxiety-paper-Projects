"""TaskWeaver编排适配器模块。

提供TaskWeaver任务编排系统的集成接口，
包括能力检查、工具目录和运行时验证。
"""

from __future__ import annotations

import importlib.util

from pi_agent.evidence_reader import READ_ONLY_KNOWLEDGE_OPERATIONS


def capability() -> dict:
    """检查TaskWeaver后端能力。

    Returns:
        包含available和reason键的能力状态字典。
    """
    available = importlib.util.find_spec("taskweaver") is not None
    return {"available": available, "reason": "taskweaver installed" if available else "taskweaver is optional and not installed"}


def tool_catalog() -> list[dict]:
    """获取已注册工具的目录。

    Returns:
        工具信息字典列表，每个字典包含name、module和contract键。
    """
    return [{"name": name, "module": "audit-passed snapshot", "contract": "dict -> read-only dict"}
            for name in sorted(READ_ONLY_KNOWLEDGE_OPERATIONS)]


def verify_runtime() -> dict:
    """验证TaskWeaver运行时环境。

    检查TaskWeaver应用入口是否可以正常导入，
    不启动LLM会话。

    Returns:
        包含运行时状态信息的字典。
    """
    """Import TaskWeaver's application entry point without starting an LLM session."""
    status = capability()
    if not status["available"]:
        return status
    try:
        from taskweaver.app.app import TaskWeaverApp  # noqa: F401
    except Exception as exc:
        return {"available": False, "reason": f"TaskWeaver import failed: {exc}"}
    return {
        "available": True,
        "reason": "TaskWeaver application runtime imports successfully",
        "registered_tools": len(tool_catalog()),
    }
