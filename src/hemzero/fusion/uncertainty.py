"""不确定性量化模块。

提供多视图预测的不确定性分析功能，
计算视图间分歧和预测稳定性指标。
"""

from __future__ import annotations

import numpy as np


def uncertainty_summary(probabilities: np.ndarray, repeated_predictions: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """计算预测不确定性摘要。

    分析多视图概率预测的不确定性，包括视图间分歧、
    预测标准差和翻转率。

    Args:
        probabilities: 多视图概率预测数组，形状为(样本数, 视图数)。
        repeated_predictions: 可选的重复预测数组，形状为(重复次数, 样本数, 视图数)。

    Returns:
        不确定性指标字典，包含：
        - view_disagreement: 视图间预测分歧（标准差）
        - prediction_std: 重复预测的标准差
        - flip_rate: 预测类别翻转率
        - uncertainty_score: 综合不确定性分数
    """
    probabilities = np.asarray(probabilities, dtype=float)
    disagreement = probabilities.std(axis=1)
    if repeated_predictions is None:
        prediction_std = np.zeros(len(probabilities))
        flip_rate = np.zeros(len(probabilities))
    else:
        repeated = np.asarray(repeated_predictions, dtype=float)
        prediction_std = repeated.std(axis=0).mean(axis=1)
        flip_rate = np.minimum((repeated >= 0.5).mean(axis=0), (repeated < 0.5).mean(axis=0)).mean(axis=1)
    return {"view_disagreement": disagreement, "prediction_std": prediction_std, "flip_rate": flip_rate,
            "uncertainty_score": np.maximum(disagreement, prediction_std)}
