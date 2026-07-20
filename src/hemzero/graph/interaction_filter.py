"""特征交互过滤模块。

基于特征谱系图的知识引导交互项过滤，
用于识别和过滤机械性冗余的特征交互。
"""

from __future__ import annotations

from collections import Counter

import networkx as nx


def is_mechanically_redundant(graph: nx.MultiDiGraph, left: str, right: str) -> bool:
    """判断两个特征是否存在机械性冗余。

    检查两个特征是否存在谱系关系或共享成分关系，
    存在这些关系则认为交互是冗余的。

    Args:
        graph: 特征谱系图。
        left: 第一个特征名称。
        right: 第二个特征名称。

    Returns:
        存在机械性冗余返回True，否则返回False。
    """
    if left == right:
        return True
    for source, target in ((left, right), (right, left)):
        relations = {data.get("relation") for data in graph.get_edge_data(source, target, default={}).values()}
        if relations & {"derived_from", "numerator_of", "denominator_of", "shared_component"}:
            return True
    return False


def filter_interactions(
    candidates: list[dict], graph: nx.MultiDiGraph, *, min_frequency: float = 0.4,
    min_direction_consistency: float = 0.6, max_interactions: int = 2,
) -> tuple[list[dict], list[dict]]:
    """过滤候选特征交互项。

    根据频率阈值、谱系冗余检查和复杂度上限过滤交互项，
    返回接受和拒绝两个列表。

    Args:
        candidates: 候选交互项列表，每个元素为包含features、frequency、score等键的字典。
        graph: 特征谱系图。
        min_frequency: 最小频率阈值，默认为0.4。
        max_interactions: 最大接受交互项数，默认为2。

    Returns:
        元组，包含接受列表和拒绝列表。
        拒绝列表中的元素包含rejection_reason字段说明拒绝原因。
    """
    accepted, rejected = [], []
    for candidate in sorted(candidates, key=lambda item: (-item.get("frequency", 0), -abs(item.get("score", 0)))):
        features = candidate.get("features", [])
        reason = None
        if len(features) != 2:
            reason = "interaction_order_not_two"
        elif candidate.get("frequency", 0) < min_frequency:
            reason = "unstable_frequency"
        elif candidate.get("direction_consistency", 1.0) < min_direction_consistency:
            reason = "unstable_direction"
        elif is_mechanically_redundant(graph, features[0], features[1]):
            reason = "lineage_redundancy"
        if reason:
            rejected.append({**candidate, "rejection_reason": reason})
        elif len(accepted) < max_interactions:
            accepted.append(candidate)
        else:
            rejected.append({**candidate, "rejection_reason": "complexity_cap"})
    return accepted, rejected
