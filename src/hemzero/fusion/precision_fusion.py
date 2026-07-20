"""精确度融合模块。

提供基于方差的加权融合方法，
用于多视图或多模型的概率预测融合。
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logit


def precision_fusion(probabilities: np.ndarray, variances: np.ndarray, gates: np.ndarray | None = None, *, epsilon: float = 1e-4):
    """执行精确度加权的概率融合。

    使用逆方差加权方法融合多个概率预测，
    支持额外的门控权重调制。

    Args:
        probabilities: 概率预测数组，形状为(样本数, 视图数)。
        variances: 预测方差数组，形状与概率数组相同。
        gates: 可选的门控权重数组，形状与概率数组相同。
        epsilon: 数值稳定性的小常数。

    Returns:
        元组，包含：
        - 融合后的概率数组
        - 使用的权重数组
    """
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    variances = np.asarray(variances, dtype=float)
    if gates is None:
        gates = np.ones_like(probabilities)
    weights = np.asarray(gates, dtype=float) / (variances + epsilon)
    fused_logit = (weights * logit(probabilities)).sum(axis=1) / np.maximum(weights.sum(axis=1), epsilon)
    return expit(fused_logit), weights
