"""EBM学生模型模块。

提供解释性提升机（EBM）模型的创建和训练功能，
支持自定义特征交互项。
"""

from __future__ import annotations


def create_ebm(*, interactions=0, seed: int = 284, max_rounds: int = 1000, outer_bags: int = 8):
    """创建解释性提升机分类器。

    Args:
        interactions: 特征交互项数量或索引元组列表，0表示无交互。
        seed: 随机种子。
        max_rounds: 最大迭代轮数。
        outer_bags: 外部袋数，用于稳定性。

    Returns:
        配置好的ExplainableBoostingClassifier实例。
    """
    from interpret.glassbox import ExplainableBoostingClassifier

    return ExplainableBoostingClassifier(
        interactions=interactions, random_state=seed, max_rounds=max_rounds,
        outer_bags=outer_bags, inner_bags=0, n_jobs=-1,
    )


def train_ebm_students(
    features, labels, selected_interactions: list[tuple[int, int]], *, seed: int = 284,
    max_rounds: int = 1000, outer_bags: int = 8,
):
    """训练多个EBM学生模型变体。

    训练无交互、自动交互和指定交互三种EBM模型，
    用于比较不同交互策略的效果。

    Args:
        features: 训练特征数据框。
        labels: 训练标签数组。
        selected_interactions: 选定的特征交互项索引元组列表。
        seed: 随机种子。

    Returns:
        包含三个训练好的EBM模型的字典：
        - 'EBM-0': 无交互项的基线模型
        - 'EBM-Auto': 自动选择少量交互项的模型
        - 'EBM-TabDistill': 使用TabDistill选定交互项的模型
    """
    models = {
        "EBM-0": create_ebm(interactions=0, seed=seed, max_rounds=max_rounds, outer_bags=outer_bags),
        "EBM-Auto": create_ebm(interactions=min(2, max(1, len(selected_interactions))), seed=seed,
                               max_rounds=max_rounds, outer_bags=outer_bags),
        "EBM-TabDistill": create_ebm(interactions=selected_interactions, seed=seed,
                                     max_rounds=max_rounds, outer_bags=outer_bags),
    }
    for model in models.values():
        model.fit(features, labels)
    return models
