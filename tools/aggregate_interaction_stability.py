from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from hemzero.common.hashing import sha256_file
from hemzero.common.schemas import ArtifactRef, Status, ToolResult
from hemzero.knowledge_graph.snapshot import read_jsonl, write_jsonl

from ._kg_shared import open_repository
from ._shared import ToolContext, require_file


def _summary(run_id: str, pair_id: str, rows: list[dict], *, repeat: int | None, fold: int | None, scope: str, total_folds: int) -> dict:
    scores = np.asarray([float(row["score"]) for row in rows])
    if scope == "fold_local":
        frequency = max(float(row["frequency"]) for row in rows)
        signed_consistency = [np.sign(float(row["score"])) * float(row["direction_consistency"]) for row in rows]
        direction_consistency = abs(float(np.mean(signed_consistency)))
    else:
        frequency = len({(row["repeat"], row["fold"]) for row in rows}) / total_folds
        direction_consistency = abs(float(np.mean(np.sign(scores))))
    suffix = f"r{repeat}__f{fold}" if repeat is not None else "global"
    first = rows[0]
    return {
        "summary_id": f"{run_id}__{suffix}__{pair_id}", "run_id": run_id, "scope": scope,
        "repeat": repeat, "fold": fold, "pair_id": pair_id,
        "features": [first["feature_a"], first["feature_b"]],
        "feature_a": first["feature_a"], "feature_b": first["feature_b"],
        "feature_a_id": first["feature_a_id"], "feature_b_id": first["feature_b_id"],
        "frequency": float(frequency), "direction_consistency": float(direction_consistency),
        "score": float(np.mean(scores)), "supporting_indices": sorted({row["interaction_index"] for row in rows}),
        "candidate_count": len(rows),
    }


def run(arguments: dict, context: ToolContext) -> ToolResult:
    source = require_file(context.run_dir / "knowledge_graph" / "interaction_candidates.jsonl", "interaction_candidates")
    candidates = read_jsonl(source)
    grouped: dict[tuple[int, int, str], list[dict]] = defaultdict(list)
    global_grouped: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        grouped[(int(row["repeat"]), int(row["fold"]), row["pair_id"])].append(row)
        global_grouped[row["pair_id"]].append(row)
    splits = pd.read_csv(require_file(context.run_dir / "splits" / "outer_split_assignment.csv", "outer_splits"))
    total_folds = len(splits[["repeat", "fold"]].drop_duplicates())
    summaries = [
        _summary(context.run_dir.name, pair_id, rows, repeat=repeat, fold=fold, scope="fold_local", total_folds=total_folds)
        for (repeat, fold, pair_id), rows in sorted(grouped.items())
    ]
    summaries.extend(
        _summary(context.run_dir.name, pair_id, rows, repeat=None, fold=None, scope="development_global", total_folds=total_folds)
        for pair_id, rows in sorted(global_grouped.items())
    )
    try:
        with open_repository(context) as repository:
            repository.upsert_summaries(summaries)
    except Exception as exc:
        return ToolResult("aggregate_interaction_stability", Status.BLOCKED, f"Stability summary persistence failed: {exc}")
    output = context.run_dir / "knowledge_graph" / "interaction_summaries.jsonl"
    write_jsonl(output, summaries)
    return ToolResult("aggregate_interaction_stability", Status.COMPLETE, "Fold-local and development-only interaction stability aggregated",
                      [ArtifactRef(str(output), sha256_file(output), "application/x-ndjson")],
                      {"fold_local_summaries": len(summaries) - len(global_grouped),
                       "global_summaries": len(global_grouped), "total_folds": total_folds,
                       "holdout_labels_accessed": False})
