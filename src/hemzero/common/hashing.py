"""哈希工具模块。

提供文件和数据的确定性哈希计算功能，
用于数据完整性校验和内容寻址。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """计算文件的SHA256哈希值。

    使用流式读取处理大文件，避免内存溢出。

    Args:
        path: 文件路径。

    Returns:
        文件的SHA256十六进制哈希字符串。
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    """计算任意值的稳定哈希值。

    将值序列化为排序后的JSON字符串后计算哈希，
    确保相同数据结构产生相同的哈希值。

    Args:
        value: 要计算哈希的值，支持任意可JSON序列化的对象。

    Returns:
        SHA256十六进制哈希字符串。
    """
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()
