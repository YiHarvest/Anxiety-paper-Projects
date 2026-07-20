"""文件I/O工具模块。

提供YAML配置加载、原子化JSON写入和路径安全验证功能，
确保文件操作的可靠性和安全性。
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    """加载YAML配置文件。

    Args:
        path: YAML文件路径。

    Returns:
        解析后的字典对象。

    Raises:
        ValueError: 文件内容不是字典类型时抛出。
    """
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    """原子化写入JSON文件。

    使用临时文件和原子替换操作，确保写入过程的原子性，
    防止写入过程中断导致的数据损坏。

    Args:
        path: 目标文件路径。
        value: 要写入的值，支持任意可JSON序列化的对象。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, default=str)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def require_within(path: Path, root: Path) -> Path:
    """验证路径是否在指定根目录内。

    用于防止路径逃逸攻击，确保文件操作不会越界。

    Args:
        path: 要验证的路径。
        root: 根目录路径。

    Returns:
        解析后的绝对路径。

    Raises:
        ValueError: 路径逃逸根目录时抛出。
    """
    resolved, resolved_root = path.resolve(), root.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"Path escapes artifact root: {path}")
    return resolved
