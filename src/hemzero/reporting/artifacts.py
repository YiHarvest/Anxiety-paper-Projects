"""制品存储管理模块。

提供实验运行目录的创建、冻结和制品清单管理功能，
确保制品的完整性和不可变性。
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from hemzero.common.io import atomic_json, require_within
from hemzero.reporting.evidence import build_manifest


ARTIFACT_SUBDIRS = (
    "dataset", "splits", "predictions", "models", "formulas", "figures", "reports", "evidence",
    "knowledge_graph",
)


class ArtifactStore:
    """制品存储管理器。

    管理实验运行目录的创建和生命周期，
    确保制品的安全存储和完整性验证。

    Attributes:
        root: 制品存储根目录路径。
    """

    def __init__(self, root: Path):
        """初始化制品存储管理器。

        Args:
            root: 制品存储根目录路径。
        """
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def create_run(self, run_id: str | None = None) -> Path:
        """创建新的运行目录。

        生成唯一的运行ID并创建标准的目录结构。

        Args:
            run_id: 可选的运行ID，未提供时自动生成时间戳ID。

        Returns:
            新创建的运行目录路径。

        Raises:
            FileExistsError: 运行ID已存在时抛出。
        """
        run_id = run_id or f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
        run_dir = require_within(self.root / run_id, self.root)
        if run_dir.exists():
            raise FileExistsError(f"Run already exists and cannot be reused: {run_id}")
        for subdir in ARTIFACT_SUBDIRS:
            (run_dir / subdir).mkdir(parents=True)
        atomic_json(run_dir / "evidence" / "run.json", {"run_id": run_id, "created_at": datetime.now(UTC).isoformat(), "frozen": False})
        return run_dir

    @staticmethod
    def assert_writable(run_dir: Path) -> None:
        """断言运行目录可写入。

        检查运行目录是否已被冻结。

        Args:
            run_dir: 运行目录路径。

        Raises:
            PermissionError: 运行目录已冻结时抛出。
        """
        if (run_dir / "_FROZEN").exists():
            raise PermissionError(f"Run is frozen: {run_dir}")

    def finalize(self, run_dir: Path) -> Path:
        """完成运行并冻结制品。

        生成制品清单并冻结运行目录，防止后续修改。

        Args:
            run_dir: 运行目录路径。

        Returns:
            制品清单文件路径。
        """
        """Freeze successful, failed, or blocked terminal runs after their final state is written."""
        self.assert_writable(run_dir)
        manifest = build_manifest(run_dir)
        manifest_path = run_dir / "manifest.json"
        atomic_json(manifest_path, manifest)
        (run_dir / "_FROZEN").touch(exist_ok=False)
        return manifest_path
