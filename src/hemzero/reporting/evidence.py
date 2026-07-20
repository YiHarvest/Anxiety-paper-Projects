"""证据清单模块。

提供实验制品清单的构建和验证功能，
用于确保制品的完整性和可追溯性。
"""

from __future__ import annotations

import json
from pathlib import Path

from hemzero.common.hashing import sha256_file


def build_manifest(run_dir: Path) -> dict:
    """构建运行目录的制品清单。

    遍历运行目录中的所有文件，生成包含路径、哈希和大小信息的清单。

    Args:
        run_dir: 运行目录路径。

    Returns:
        制品清单字典，包含run_id、immutable标志和files列表。
    """
    files = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append({"path": str(path.relative_to(run_dir)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return {"run_id": run_dir.name, "immutable": True, "files": files}


def verify_manifest(run_dir: Path, manifest: dict) -> list[str]:
    """验证制品清单与实际文件的一致性。

    检查清单中列出的文件是否存在且哈希值是否匹配。

    Args:
        run_dir: 运行目录路径。
        manifest: 制品清单字典。

    Returns:
        错误信息列表，包含缺失文件和哈希不匹配的文件。
    """
    errors = []
    for item in manifest.get("files", []):
        path = run_dir / item["path"]
        if not path.exists():
            errors.append(f"missing:{item['path']}")
        elif sha256_file(path) != item["sha256"]:
            errors.append(f"hash_mismatch:{item['path']}")
    return errors
