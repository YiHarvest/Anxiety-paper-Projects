"""概率校准模块。

提供Logit校准方法，用于校准模型的概率输出，
使预测概率更准确地反映真实风险。
"""

from __future__ import annotations

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression


def fit_logit_calibrator(oof_probability: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    """拟合Logit校准器。

    使用Logit变换后的概率作为特征，训练逻辑回归模型
    进行概率校准。

    Args:
        oof_probability: 袋外预测概率数组。
        labels: 真实标签数组。

    Returns:
        训练好的Logit校准器模型。
    """
    values = logit(np.clip(oof_probability, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    return LogisticRegression(C=1e6, max_iter=2000).fit(values, labels)


def apply_logit_calibrator(model: LogisticRegression, probability: np.ndarray) -> np.ndarray:
    """应用Logit校准器。

    对预测概率进行Logit校准变换。

    Args:
        model: 已训练的Logit校准器模型。
        probability: 原始预测概率数组。

    Returns:
        校准后的预测概率数组。
    """
    values = logit(np.clip(probability, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    return model.predict_proba(values)[:, 1]
