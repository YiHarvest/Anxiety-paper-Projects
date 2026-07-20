"""多视图构建模块。

提供特征视图构建功能，支持原始特征视图、
比率特征视图和完整视图的分离构建。
"""

from __future__ import annotations

import pandas as pd


def build_views(frame: pd.DataFrame, raw: list[str], ratios: list[str]) -> dict[str, pd.DataFrame]:
    """构建多个特征视图。

    根据特征列表构建原始特征视图、比率特征视图和完整视图，
    用于多视图学习或特征分析。

    Args:
        frame: 包含所有特征的数据框。
        raw: 原始特征列名列表。
        ratios: 比率特征列名列表。

    Returns:
        字典，包含三个视图：
        - 'raw': 仅包含原始特征的视图
        - 'ratio': 仅包含比率特征的视图
        - 'full': 包含所有特征的完整视图

    Raises:
        ValueError: 指定的特征列不存在时抛出。
    """
    missing = sorted(set(raw + ratios) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing view columns: {missing}")
    return {"raw": frame[raw].copy(), "ratio": frame[ratios].copy(), "full": frame[raw + ratios].copy()}
