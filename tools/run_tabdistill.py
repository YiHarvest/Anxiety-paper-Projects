from __future__ import annotations

from pathlib import Path

import pandas as pd

from hemzero.common.hashing import sha256_file
from hemzero.common.schemas import ArtifactRef, Status, ToolResult
from hemzero.distillation.tabdistill_runner import capability, extract_interactions_from_data
from hemzero.knowledge_graph.lineage_sync import canonical_pair, feature_entity_id
from hemzero.knowledge_graph.snapshot import write_jsonl

from ._kg_shared import open_repository
from ._shared import ToolContext, require_file


def run(arguments: dict, context: ToolContext) -> ToolResult:
    dataset, experiment, models = context.config("dataset.yaml"), context.config("experiment.yaml"), context.config("models.yaml")
    repository_path = Path(arguments.get("repository", models["tabdistill"]["repository"]))
    backend = capability(repository_path)
    if not backend.available:
        return ToolResult("run_tabdistill", Status.BLOCKED, backend.reason)
    execution = experiment.get("execution", {})
    search_repeats = int(arguments.get("search_repeats", execution.get("tabdistill_repeats", 1)))
    folds_per_repeat = int(arguments.get("folds_per_repeat", execution.get("tabdistill_folds_per_repeat", 5)))
    max_samples = int(arguments.get("max_samples", execution.get("tabdistill_max_samples", 3)))
    frame = pd.read_csv(require_file(context.run_dir / "dataset" / "cleaned_train.csv", "cleaned_train"))
    splits = pd.read_csv(require_file(context.run_dir / "splits" / "outer_split_assignment.csv", "outer_splits"))
    features = dataset["raw_features"] + dataset["ratio_features"]
    indexed = frame.set_index(dataset["id_column"])
    candidates: list[dict] = []
    for repeat in sorted(splits.repeat.unique())[:search_repeats]:
        repeated = splits[splits.repeat == repeat]
        for fold in sorted(repeated.fold.unique())[:folds_per_repeat]:
            partition = repeated[repeated.fold == fold]
            training_path = context.run_dir / "dataset" / f"tabdistill_outer_train_r{repeat}_f{fold}.csv"
            train_ids = partition[partition.split == "train"].patient_id.tolist()
            indexed.loc[train_ids, features + [dataset["target_column"]]].to_csv(training_path, index=False)
            payload_path = context.run_dir / "evidence" / f"tabdistill_candidates_r{repeat}_f{fold}.json"
            payload = extract_interactions_from_data(
                repository=repository_path, training_csv=training_path, output_json=payload_path,
                feature_names=features, target_column=dataset["target_column"],
                sample_budget=int(models["tabdistill"]["sample_budget"]), max_samples=max_samples,
                index_types=list(models["tabdistill"]["index_types"]),
                seed=int(experiment["seed"]) + 100 * int(repeat) + int(fold),
            )
            by_index: dict[str, list[dict]] = {}
            for item in payload.get("candidates", []):
                by_index.setdefault(str(item["index_type"]), []).append(item)
            for index_type, items in by_index.items():
                ranked = sorted(items, key=lambda row: -abs(float(row.get("score", 0))))
                for rank, item in enumerate(ranked, start=1):
                    left, right = sorted(map(str, item["features"]), key=feature_entity_id)
                    pair_id = canonical_pair(left, right)
                    candidates.append({
                        "candidate_id": f"{context.run_dir.name}__r{int(repeat)}__f{int(fold)}__{pair_id}__{index_type}",
                        "run_id": context.run_dir.name, "repeat": int(repeat), "fold": int(fold),
                        "fold_id": f"{context.run_dir.name}__r{int(repeat)}__f{int(fold)}",
                        "pair_id": pair_id, "feature_a": left, "feature_b": right,
                        "feature_a_id": feature_entity_id(left), "feature_b_id": feature_entity_id(right),
                        "interaction_index": index_type, "score": float(item.get("score", 0)),
                        "interaction_value": float(item.get("score", 0)),
                        "direction": "positive" if float(item.get("score", 0)) > 0 else "negative" if float(item.get("score", 0)) < 0 else "zero",
                        "frequency": float(item.get("frequency", 0)),
                        "direction_consistency": float(item.get("direction_consistency", 0)),
                        "rank": rank, "teacher_backend": "TabPFN-3", "source_partition": "outer_training_fold",
                        "uses_holdout_labels": False,
                    })
    deduplicated: dict[str, dict] = {}
    for row in candidates:
        current = deduplicated.get(row["candidate_id"])
        if current is None or abs(row["score"]) > abs(current["score"]):
            deduplicated[row["candidate_id"]] = row
    candidates = list(deduplicated.values())
    try:
        with open_repository(context) as repository:
            repository.upsert_candidates(candidates)
    except Exception as exc:
        return ToolResult("run_tabdistill", Status.BLOCKED, f"Candidate persistence failed: {exc}")
    output = context.run_dir / "knowledge_graph" / "interaction_candidates.jsonl"
    write_jsonl(output, candidates)
    return ToolResult("run_tabdistill", Status.COMPLETE, "Every fold-local TabDistill candidate persisted",
                      [ArtifactRef(str(output), sha256_file(output), "application/x-ndjson")],
                      {"backend": "Clouddelta/tab-distill", "teacher_backend": "TabPFN-3",
                       "candidates": len(candidates), "fold_local": True,
                       "search_repeats_executed": search_repeats,
                       "search_folds_per_repeat": folds_per_repeat,
                       "test_labels_accessed": False})
