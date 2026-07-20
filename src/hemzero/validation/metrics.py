"""评估指标模块。

提供分类模型的综合评估指标计算功能，
包括AUROC、AUPRC、Brier分数、F1等。
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, f1_score, log_loss, matthews_corrcoef, roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.special import logit


def classification_metrics(labels, probability, *, threshold: float = 0.5) -> dict:
    """计算分类模型的综合评估指标。

    根据预测概率和阈值计算多种评估指标，
    包括判别能力、校准效果和分类性能指标。

    Args:
        labels: 真实标签数组。
        probability: 预测概率数组。
        threshold: 分类阈值，默认为0.5。

    Returns:
        包含以下指标的字典：
        - AUROC: ROC曲线下面积
        - AUPRC: PR曲线下面积
        - Brier: Brier分数
        - LogLoss: 对数损失
        - Sensitivity: 敏感度（召回率）
        - Specificity: 特异度
        - F1: F1分数
        - MCC: 马修斯相关系数
        - threshold: 使用的阈值
    """
    probability = np.asarray(probability, dtype=float)
    if not np.isfinite(probability).all():
        raise ValueError("Classification probabilities must be finite")
    probability = np.clip(probability, 1e-8, 1 - 1e-8)
    prediction = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    calibration = LogisticRegression(C=1e6, max_iter=2000).fit(logit(probability).reshape(-1, 1), labels)
    return {
        "AUROC": roc_auc_score(labels, probability), "AUPRC": average_precision_score(labels, probability),
        "Brier": brier_score_loss(labels, probability), "LogLoss": log_loss(labels, probability),
        "Sensitivity": tp / max(tp + fn, 1), "Specificity": tn / max(tn + fp, 1),
        "F1": f1_score(labels, prediction, zero_division=0), "MCC": matthews_corrcoef(labels, prediction),
        "CalibrationIntercept": float(calibration.intercept_[0]),
        "CalibrationSlope": float(calibration.coef_[0, 0]),
        "threshold": threshold,
    }
