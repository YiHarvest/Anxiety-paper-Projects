"""文稿生成模块。

提供研究结果文稿的辅助生成功能，
生成结构化的结果摘要文本。
"""

from __future__ import annotations


def results_stub(run_id: str, *, backend: str, caveats: list[str]) -> str:
    """生成研究结果摘要文稿。

    创建包含运行标识、教师后端和证据限制的结构化摘要。

    Args:
        run_id: 运行标识符。
        backend: 教师模型后端名称。
        caveats: 证据限制说明列表。

    Returns:
        Markdown格式的结果摘要文本。
    """
    caveat_text = "\n".join(f"- {item}" for item in caveats)
    return f"# HemoZero run {run_id}\n\nTeacher backend: {backend}.\n\n## Evidence limitations\n\n{caveat_text}\n"
