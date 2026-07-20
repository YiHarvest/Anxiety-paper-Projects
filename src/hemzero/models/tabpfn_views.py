"""TabPFN多视图模型模块。

提供TabPFN基础模型的适配器和多视图训练功能，
支持在原始特征、比率特征和完整视图上训练模型。
"""

from __future__ import annotations

import importlib.util
import gc
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import softmax


@dataclass(frozen=True)
class BackendCapability:
    """后端能力检查结果数据类。

    表示可选后端模块的可用性状态。

    Attributes:
        available: 后端是否可用。
        reason: 不可用原因或可用说明。
        authenticated: 是否已完成认证配置。
    """

    available: bool
    reason: str
    authenticated: bool


def tabpfn_capability() -> BackendCapability:
    """检查TabPFN后端能力。

    检查tabpfn包是否安装以及认证配置状态。

    Returns:
        后端能力检查结果。
    """
    if importlib.util.find_spec("tabpfn") is None:
        return BackendCapability(False, "tabpfn package is not installed; use uv sync --extra foundation", False)
    token = bool(os.getenv("TABPFN_TOKEN"))
    cache = bool(os.getenv("TABPFN_MODEL_CACHE_DIR"))
    return BackendCapability(
        True,
        "tabpfn package detected; checkpoint access is verified when the model is initialized",
        token or cache,
    )


def create_tabpfn_classifier(
    *, device: str = "cpu", seed: int = 284, n_estimators: int = 4
) -> Any:
    """创建TabPFN分类器实例。

    Args:
        device: 计算设备，默认为'cpu'。
        seed: 随机种子。
        n_estimators: 估计器数量。

    Returns:
        TabPFN分类器实例。

    Raises:
        RuntimeError: TabPFN包未安装时抛出。
    """
    capability = tabpfn_capability()
    if not capability.available:
        raise RuntimeError(capability.reason)
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier(
        device=device,
        random_state=seed,
        n_estimators=n_estimators,
        auto_scale_n_estimators=False,
        fit_mode="low_memory",
        memory_saving_mode=True,
        keep_cache_on_device=False,
        n_preprocessing_jobs=1,
        show_progress_bar=False,
    )


def repeated_fit_predict(
    train_features,
    train_labels,
    evaluation_features,
    *,
    device: str = "cpu",
    seed: int = 284,
    prediction_seeds: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """使用独立种子多次训练TabPFN并估计概率和认知不确定性。

    通过多次独立训练获取预测分布，用于估计模型不确定性。

    Args:
        train_features: 训练特征。
        train_labels: 训练标签。
        evaluation_features: 评估特征。
        device: 计算设备，默认为'cpu'。
        seed: 基础随机种子。
        prediction_seeds: 独立训练次数。

    Returns:
        元组，包含平均预测概率数组和认知方差数组。
    """
    model = create_tabpfn_classifier(
        device=device, seed=seed, n_estimators=prediction_seeds
    )
    model.fit(train_features, train_labels)
    if prediction_seeds > 1:
        raw_logits = model.predict_raw_logits(evaluation_features)
        temperature = float(model.softmax_temperature_)
        matrix = softmax(raw_logits / temperature, axis=-1)[:, :, 1]
    else:
        matrix = model.predict_proba(evaluation_features)[:, 1][None, :]
    del model
    gc.collect()
    variance = matrix.var(axis=0, ddof=1) if len(matrix) > 1 else np.full(matrix.shape[1], 1e-4)
    return matrix.mean(axis=0), np.maximum(variance, 1e-6)


def train_three_views(
    views: dict,
    labels,
    *,
    device: str = "cpu",
    seed: int = 284,
    n_estimators: int = 4,
) -> dict:
    """在三个视图上训练TabPFN模型。

    分别在原始特征、比率特征和完整视图上训练独立的TabPFN模型。

    Args:
        views: 视图字典，必须包含'raw'、'ratio'和'full'三个视图。
        labels: 训练标签。
        device: 计算设备，默认为'cpu'。
        seed: 随机种子。
        n_estimators: 估计器数量。

    Returns:
        包含三个训练好的模型的字典，键为视图名称。

    Raises:
        ValueError: 视图字典不包含所需的三个视图时抛出。
    """
    required = {"raw", "ratio", "full"}
    if set(views) != required:
        raise ValueError(f"Expected exactly {sorted(required)}, got {sorted(views)}")
    models = {}
    for offset, name in enumerate(("raw", "ratio", "full")):
        model = create_tabpfn_classifier(
            device=device, seed=seed + offset, n_estimators=n_estimators
        )
        model.fit(views[name], labels)
        models[name] = model
    return models
