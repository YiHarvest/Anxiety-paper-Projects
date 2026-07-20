"""Bootstrap置信区间模块。

提供基于Bootstrap的置信区间估计功能，
用于量化模型性能指标的统计不确定性。
"""

from __future__ import annotations

import numpy as np

from .metrics import classification_metrics


def bootstrap_intervals(labels, probability, *, threshold: float, n_resamples: int = 500, seed: int = 284) -> list[dict]:
    """计算模型性能指标的Bootstrap置信区间。

    通过重采样方法估计各指标的95%置信区间。

    Args:
        labels: 真实标签数组。
        probability: 预测概率数组。
        threshold: 分类阈值。
        n_resamples: Bootstrap重采样次数，默认为500。
        seed: 随机种子。

    Returns:
        各指标的置信区间结果列表，每个元素包含：
        - metric: 指标名称
        - estimate: 点估计值
        - ci_lower: 置信区间下限
        - ci_upper: 置信区间上限
        - valid_resamples: 有效重采样次数
    """
    labels, probability = np.asarray(labels), np.asarray(probability)
    rng, samples = np.random.default_rng(seed), []
    for _ in range(n_resamples):
        indices = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[indices])) == 2:
            samples.append(classification_metrics(labels[indices], probability[indices], threshold=threshold))
    point = classification_metrics(labels, probability, threshold=threshold)
    return [
        {"metric": name, "estimate": point[name], "ci_lower": np.quantile([row[name] for row in samples], 0.025),
         "ci_upper": np.quantile([row[name] for row in samples], 0.975), "valid_resamples": len(samples)}
        for name in ("AUROC", "AUPRC", "Brier", "LogLoss", "Sensitivity", "Specificity", "F1", "MCC")
    ]
