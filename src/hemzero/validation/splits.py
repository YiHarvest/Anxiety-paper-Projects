"""交叉验证拆分模块。

提供患者级别的无泄漏交叉验证拆分功能，
确保训练集和测试集的患者不重叠。
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold


def create_repeated_splits(ids, labels, *, n_splits: int = 5, n_repeats: int = 5, seed: int = 284) -> pd.DataFrame:
    """创建重复分层K折交叉验证拆分表。

    生成患者级别的拆分分配表，支持多次重复的分层拆分。

    Args:
        ids: 患者ID数组。
        labels: 标签数组，用于分层。
        n_splits: 折数，默认为5。
        n_repeats: 重复次数，默认为5。
        seed: 随机种子。

    Returns:
        拆分分配表，包含repeat、fold、patient_id和split列。
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    rows = []
    for run, (train_indices, validation_indices) in enumerate(cv.split(ids, labels)):
        repeat, fold = divmod(run, n_splits)
        rows.extend({"repeat": repeat, "fold": fold, "patient_id": ids[index], "split": "train"} for index in train_indices)
        rows.extend({"repeat": repeat, "fold": fold, "patient_id": ids[index], "split": "validation"} for index in validation_indices)
    return pd.DataFrame(rows)


def assert_disjoint(train_ids, test_ids) -> None:
    """断言训练集和测试集患者ID不相交。

    用于检测数据泄漏，确保模型评估的有效性。

    Args:
        train_ids: 训练集患者ID集合或列表。
        test_ids: 测试集患者ID集合或列表。

    Raises:
        ValueError: 发现重叠患者ID时抛出。
    """
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise ValueError(f"Patient leakage detected: {sorted(overlap)[:5]}")
