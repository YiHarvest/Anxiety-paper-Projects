"""嵌套交叉验证模块。

提供嵌套交叉验证的袋外预测和折叠级回调功能，
确保内部验证与外部测试的严格分离。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold


def inner_oof_predict(estimator, features, labels, *, n_splits: int = 4, seed: int = 284) -> np.ndarray:
    """在内部训练分区中生成袋外预测。

    严格限制在外部训练分区内进行交叉验证，
    生成用于模型选择或超参数调优的软标签。

    Args:
        estimator: 估计器对象，需支持fit和predict_proba方法。
        features: 特征数据框。
        labels: 标签数组。
        n_splits: 内部交叉验证折数，默认为4。
        seed: 随机种子。

    Returns:
        袋外预测概率数组。
    """
    """Generate soft labels strictly inside one outer-training partition."""
    labels = np.asarray(labels)
    prediction = np.zeros(len(labels), dtype=float)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_indices, validation_indices in splitter.split(features, labels):
        model = clone(estimator)
        model.fit(features.iloc[train_indices], labels[train_indices])
        prediction[validation_indices] = model.predict_proba(features.iloc[validation_indices])[:, 1]
    return prediction


def outer_fold_apply(split_table, frame, *, id_column: str, callback: Callable):
    """在拆分表的外部折叠上应用回调函数。

    按照拆分表的定义，为每个重复和折叠调用回调函数，
    传递训练集和验证集的行索引。

    Args:
        split_table: 拆分分配表，包含repeat、fold、patient_id和split列。
        frame: 数据框，包含id_column指定的ID列。
        id_column: ID列名称。
        callback: 回调函数，接收训练集索引和验证集索引。

    Yields:
        元组(repeat, fold, callback_result)，包含重复编号、折叠编号和回调返回值。
    """
    """Yield callback results without allowing access to an independent test table."""
    index = {patient_id: row for row, patient_id in enumerate(frame[id_column])}
    for (repeat, fold), partition in split_table.groupby(["repeat", "fold"]):
        train = np.array([index[value] for value in partition[partition.split == "train"].patient_id])
        validation = np.array([index[value] for value in partition[partition.split == "validation"].patient_id])
        yield repeat, fold, callback(train, validation)
