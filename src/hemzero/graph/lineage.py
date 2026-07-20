"""特征谱系图构建模块。

构建原始特征与派生特征之间的血缘关系图，
用于分析特征依赖关系和识别冗余交互。
"""

from __future__ import annotations

import networkx as nx


def build_lineage_graph(raw_features: list[str], formulas: dict[str, dict[str, str]]) -> nx.MultiDiGraph:
    """构建特征谱系有向图。

    根据原始特征列表和比率公式构建血缘关系图，
    节点表示特征，边表示推导关系和共享成分关系。

    Args:
        raw_features: 原始特征名称列表。
        formulas: 比率公式字典，键为比率名称，值为包含numerator和denominator的字典。

    Returns:
        多重有向图，节点包含kind属性（'raw'或'derived'），
        边包含relation属性表示关系类型。
    """
    graph = nx.MultiDiGraph()
    graph.add_nodes_from((feature, {"kind": "raw"}) for feature in raw_features)
    for ratio, formula in formulas.items():
        numerator, denominator = formula["numerator"], formula["denominator"]
        graph.add_node(ratio, kind="derived", formula=f"{numerator}/{denominator}")
        graph.add_edge(numerator, ratio, relation="numerator_of")
        graph.add_edge(denominator, ratio, relation="denominator_of")
        graph.add_edge(numerator, ratio, relation="derived_from")
        graph.add_edge(denominator, ratio, relation="derived_from")
    ratios = list(formulas)
    for index, left in enumerate(ratios):
        left_parts = {formulas[left]["numerator"], formulas[left]["denominator"]}
        for right in ratios[index + 1 :]:
            shared = sorted(left_parts & {formulas[right]["numerator"], formulas[right]["denominator"]})
            if shared:
                graph.add_edge(left, right, relation="shared_component", components=shared)
                graph.add_edge(right, left, relation="shared_component", components=shared)
    return graph


def graph_payload(graph: nx.MultiDiGraph) -> dict:
    """将图转换为可序列化的字典格式。

    Args:
        graph: NetworkX图对象。

    Returns:
        符合node-link数据格式的字典。
    """
    return nx.node_link_data(graph, edges="edges")
