"""数据结构定义模块。

定义实验流程中使用的核心数据结构，
包括状态枚举、工具结果和实验状态等。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class Status(StrEnum):
    """实验阶段状态枚举。

    定义工具执行结果的五种状态：
    - PENDING: 待执行
    - RUNNING: 执行中
    - COMPLETE: 已完成
    - BLOCKED: 被阻塞
    - FAILED: 执行失败
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass(frozen=True)
class ArtifactRef:
    """制品引用数据类。

    表示实验过程中生成的文件制品引用。

    Attributes:
        path: 制品文件路径。
        sha256: 文件SHA256哈希值。
        media_type: MIME类型，默认为二进制流。
    """

    path: str
    sha256: str
    media_type: str = "application/octet-stream"


@dataclass
class ToolResult:
    """工具执行结果数据类。

    封装工具执行的结果信息，包括状态、摘要、制品和证据。

    Attributes:
        tool: 工具名称。
        status: 执行状态。
        summary: 执行结果摘要。
        artifacts: 生成的制品列表。
        evidence: 证据数据字典。
        error: 错误信息，执行失败时填写。
    """

    tool: str
    status: Status
    summary: str
    artifacts: list[ArtifactRef] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式。

        Returns:
            包含所有字段的字典对象。
        """
        return asdict(self)


@dataclass
class ExperimentState:
    """实验状态数据类。

    记录实验运行的完整状态信息。

    Attributes:
        run_id: 运行标识符。
        run_dir: 运行目录路径。
        completed: 已完成的阶段列表。
        blocked: 被阻塞的阶段字典，键为阶段名，值为原因。
        failed: 已失败的阶段字典，键为阶段名，值为错误信息。
        current_stage: 当前执行阶段。
        artifact_manifest: 制品清单文件路径。
    """

    run_id: str
    run_dir: str
    completed: list[str] = field(default_factory=list)
    blocked: dict[str, str] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    current_stage: str | None = None
    artifact_manifest: str | None = None

    @property
    def path(self) -> Path:
        """获取运行目录路径对象。

        Returns:
            运行目录的Path对象。
        """
        return Path(self.run_dir)
