from __future__ import annotations

import json

import networkx as nx
import pandas as pd

from hemzero.common.hashing import sha256_file
from hemzero.common.io import atomic_json
from hemzero.common.schemas import ArtifactRef, Status, ToolResult
from hemzero.graph.lineage import build_lineage_graph, graph_payload

from ._shared import ToolContext


def run(arguments: dict, context: ToolContext) -> ToolResult:
    dataset, formulas = context.config("dataset.yaml"), context.config("ratio_formulas.yaml")
    graph = build_lineage_graph(dataset["raw_features"], formulas)
    graph_path = context.run_dir / "dataset" / "feature_lineage_graph.json"
    edges_path = context.run_dir / "dataset" / "feature_lineage_edges.csv"
    atomic_json(graph_path, graph_payload(graph))
    pd.DataFrame(
        {"source": source, "target": target, **data} for source, target, data in graph.edges(data=True)
    ).to_csv(edges_path, index=False)
    evidence = {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(), "dag": nx.is_directed_acyclic_graph(graph)}
    return ToolResult("build_lineage_graph", Status.COMPLETE, "NetworkX feature lineage graph created",
                      [ArtifactRef(str(path), sha256_file(path)) for path in (graph_path, edges_path)], evidence)
