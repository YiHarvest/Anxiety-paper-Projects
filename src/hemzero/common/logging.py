"""日志配置模块。

提供JSON格式的结构化日志输出功能，
便于日志聚合和自动化分析。
"""

from __future__ import annotations

import json
import logging


class JsonFormatter(logging.Formatter):
    """JSON格式日志格式化器。

    将日志记录格式化为JSON对象，包含级别、名称和消息字段。
    """

    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录为JSON字符串。

        Args:
            record: 日志记录对象。

        Returns:
            JSON格式的日志字符串。
        """
        return json.dumps(
            {"level": record.levelname, "logger": record.name, "message": record.getMessage()},
            ensure_ascii=False,
        )


def configure_logging(level: int = logging.INFO) -> None:
    """配置结构化日志输出。

    设置JSON格式的流处理器，适用于容器化环境。

    Args:
        level: 日志级别，默认为INFO。
    """
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)
