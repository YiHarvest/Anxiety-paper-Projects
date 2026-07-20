"""预测稳定性分析模块。

提供多次预测结果的稳定性分析功能，
用于评估模型预测的一致性和可靠性。
"""

from __future__ import annotations

import numpy as np


def prediction_stability(repeated_probability: np.ndarray, *, threshold: float = 0.5) -> dict:
    """计算多次预测结果的稳定性指标。

    分析重复预测的概率分布，计算均值、标准差和翻转率。

    Args:
        repeated_probability: 多次预测概率数组，形状为(重复次数, 样本数)。
        threshold: 分类阈值，默认为0.5。

    Returns:
        稳定性指标字典，包含：
        - mean: 平均预测概率
        - std: 预测概率标准差
        - flip_rate: 翻转率，表示预测类别不稳定样本的比例
    """
    values = np.asarray(repeated_probability, dtype=float)
    positive_fraction = (values >= threshold).mean(axis=0)
    return {"mean": values.mean(axis=0), "std": values.std(axis=0),
            "flip_rate": np.minimum(positive_fraction, 1 - positive_fraction)}
