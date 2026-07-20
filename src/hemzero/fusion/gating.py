"""质量门控模块。

提供数据质量和模型稳定性门控函数，
用于在模型融合时对预测质量进行加权。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def data_quality_gate(train: pd.DataFrame, evaluation: pd.DataFrame) -> np.ndarray:
    """计算基于数据质量的门控权重。

    根据评估数据相对于训练数据的分布偏差、
    越界比例和缺失率计算质量分数。

    Args:
        train: 训练特征数据框。
        evaluation: 评估特征数据框。

    Returns:
        质量门控权重数组，值域为[0, 1]。
    """
    median = train.median()
    iqr = (train.quantile(0.75) - train.quantile(0.25)).replace(0, 1)
    robust_distance = ((evaluation - median) / iqr).abs().max(axis=1).to_numpy()
    outside = ((evaluation < train.min()) | (evaluation > train.max())).mean(axis=1).to_numpy()
    missing = evaluation.isna().mean(axis=1).to_numpy()
    return np.clip(np.exp(-np.clip(robust_distance - 3, 0, None) / 5) * (1 - 0.5 * outside) * (1 - missing), 0, 1)


def stability_gate(variance: np.ndarray, *, scale: float | None = None) -> np.ndarray:
    """计算基于稳定性的门控权重。

    根据预测方差计算稳定性分数，方差越大权重越低。

    Args:
        variance: 预测方差数组。
        scale: 归一化尺度参数，未提供时使用中位数。

    Returns:
        稳定性门控权重数组，值域为[0, 1]。
    """
    variance = np.asarray(variance, dtype=float)
    reference = scale if scale is not None else max(float(np.nanmedian(variance)), 1e-6)
    return np.exp(-variance / reference)
