"""TabDistill运行器模块。

提供TabDistill特征交互提取功能的Python接口，
通过子进程调用独立的Worker脚本实现功能隔离。
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TabDistillCapability:
    """TabDistill能力检查结果数据类。

    Attributes:
        available: TabDistill是否可用。
        reason: 不可用原因或可用说明。
        repository: TabDistill代码库路径。
    """

    available: bool
    reason: str
    repository: str


def capability(repository: Path) -> TabDistillCapability:
    """检查TabDistill后端能力。

    验证TabDistill代码库是否存在所需的核心文件。

    Args:
        repository: TabDistill代码库根目录路径。

    Returns:
        TabDistill能力检查结果。
    """
    required = [repository / "src" / "spectralexplain" / "explainer.py", repository / "pyproject.toml"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        return TabDistillCapability(False, f"Missing TabDistill files: {missing}", str(repository))
    return TabDistillCapability(True, "Clouddelta/tab-distill source tree detected", str(repository))


def extract_interactions(
    *, repository: Path, teacher_model: Path, features_csv: Path, output_json: Path,
    feature_names: list[str], sample_budget: int = 1000, max_samples: int = 50, index_type: str = "fbii"
) -> dict:
    """从预训练教师模型中提取特征交互。

    使用TabDistill分析教师模型的特征交互重要性。

    Args:
        repository: TabDistill代码库路径。
        teacher_model: 教师模型文件路径。
        features_csv: 特征数据CSV文件路径。
        output_json: 输出JSON文件路径。
        feature_names: 特征名称列表。
        sample_budget: 采样预算，默认为1000。
        max_samples: 最大样本数，默认为50。
        index_type: 交互索引类型，默认为'fbii'。

    Returns:
        包含特征交互信息的字典。

    Raises:
        RuntimeError: TabDistill不可用或执行失败时抛出。
    """
    status = capability(repository)
    if not status.available:
        raise RuntimeError(status.reason)
    command = [
        sys.executable, str(Path(__file__).resolve().parents[3] / "scripts" / "tabdistill_worker.py"),
        "--repository", str(repository), "--teacher-model", str(teacher_model),
        "--features", str(features_csv), "--output", str(output_json),
        "--feature-names", json.dumps(feature_names), "--sample-budget", str(sample_budget),
        "--max-samples", str(max_samples), "--index-type", index_type,
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"TabDistill worker failed: {completed.stderr[-2000:]}")
    return json.loads(output_json.read_text(encoding="utf-8"))


def extract_interactions_from_data(
    *,
    repository: Path,
    training_csv: Path,
    output_json: Path,
    feature_names: list[str],
    target_column: str,
    sample_budget: int = 1000,
    max_samples: int = 3,
    index_types: list[str] | None = None,
    seed: int = 284,
) -> dict:
    """从训练数据中提取特征交互。

    在TabDistill Worker内部训练TabPFN教师模型，
    然后提取特征交互信息。

    Args:
        repository: TabDistill代码库路径。
        training_csv: 训练数据CSV文件路径。
        output_json: 输出JSON文件路径。
        feature_names: 特征名称列表。
        target_column: 目标列名称。
        sample_budget: 采样预算，默认为1000。
        max_samples: 最大样本数，默认为3。
        index_type: 交互索引类型，默认为'fbii'。
        seed: 随机种子。

    Returns:
        包含特征交互信息的字典。

    Raises:
        RuntimeError: TabDistill不可用或执行失败时抛出。
    """
    """Train a fold-local TabPFN teacher inside the isolated TabDistill worker."""
    status = capability(repository)
    if not status.available:
        raise RuntimeError(status.reason)
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[3] / "scripts" / "tabdistill_worker.py"),
        "--repository", str(repository),
        "--training-data", str(training_csv),
        "--target-column", target_column,
        "--output", str(output_json),
        "--feature-names", json.dumps(feature_names),
        "--sample-budget", str(sample_budget),
        "--max-samples", str(max_samples),
        "--index-types", json.dumps(index_types or ["fbii"]),
        "--seed", str(seed),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"TabDistill worker failed: {completed.stderr[-4000:]}")
    return json.loads(output_json.read_text(encoding="utf-8"))
