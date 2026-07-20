"""数据检查模块。

提供数据框结构和质量检查功能，
生成数据摘要报告用于数据验证。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def inspect_frame(frame: pd.DataFrame, *, id_column: str, target_column: str) -> dict:
    """检查数据框的基本结构和质量。

    生成包含行数、列数、重复ID、目标变量分布和缺失值等信息的摘要。

    Args:
        frame: 要检查的数据框。
        id_column: ID列名称。
        target_column: 目标变量列名称。

    Returns:
        包含数据检查结果的字典。

    Raises:
        ValueError: ID列或目标列不存在时抛出。
    """
    if id_column not in frame or target_column not in frame:
        raise ValueError("ID or target column missing")
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "duplicate_ids": int(frame[id_column].duplicated().sum()),
        "target_counts": frame[target_column].value_counts(dropna=False).sort_index().to_dict(),
        "missing": frame.isna().sum().to_dict(),
        "non_finite_numeric": {
            column: int((~np.isfinite(frame[column])).sum())
            for column in frame.select_dtypes(include="number").columns
        },
    }
