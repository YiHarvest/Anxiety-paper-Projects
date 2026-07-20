from __future__ import annotations

import pandas as pd

from hemzero.common.hashing import sha256_file
from hemzero.common.schemas import ArtifactRef, Status, ToolResult
from hemzero.knowledge_graph.cohort_graph import build_fold_associations
from hemzero.knowledge_graph.snapshot import write_jsonl

from ._kg_shared import open_repository
from ._shared import ToolContext, require_file


def run(arguments: dict, context: ToolContext) -> ToolResult:
    dataset = context.config("dataset.yaml")
    biological = context.config("biological_entities.yaml")
    kg = context.config("knowledge_graph.yaml")["cohort_associations"]
    target_diseases = [
        row for row in biological["diseases"]
        if row.get("outcome_column") == dataset["target_column"]
    ]
    if len(target_diseases) != 1:
        return ToolResult(
            "kg_build_fold_associations", Status.BLOCKED,
            "Dataset target must map to exactly one configured Disease node",
            evidence={"target_column": dataset["target_column"], "matches": len(target_diseases)},
        )
    target_disease = target_diseases[0]
    frame = pd.read_csv(require_file(context.run_dir / "dataset" / "cleaned_train.csv", "cleaned_train"))
    splits = pd.read_csv(require_file(context.run_dir / "splits" / "outer_split_assignment.csv", "outer_splits"))
    features = dataset["raw_features"] + dataset["ratio_features"]
    indexed = frame.set_index(dataset["id_column"])
    rows: list[dict] = []
    folds: list[dict] = []
    for (repeat, fold), partition in splits.groupby(["repeat", "fold"], sort=True):
        train_ids = partition[partition.split == "train"].patient_id.tolist()
        fold_id = f"{context.run_dir.name}__r{int(repeat)}__f{int(fold)}"
        folds.append({"fold_id": fold_id, "run_id": context.run_dir.name, "repeat": int(repeat),
                      "fold": int(fold), "training_rows": len(train_ids), "labels_from_holdout": False})
        rows.extend(build_fold_associations(
            indexed.loc[train_ids, features], run_id=context.run_dir.name, repeat=int(repeat), fold=int(fold),
            spearman_min_absolute=float(kg["spearman_min_absolute"]),
            partial_min_absolute=float(kg["partial_min_absolute"]),
            graphical_lasso_alpha=float(kg["graphical_lasso_alpha"]),
            epsilon=float(kg["numerical_epsilon"]),
        ))
    try:
        with open_repository(context) as repository:
            repository.upsert_run_and_folds({
                "run_id": context.run_dir.name,
                "workflow": "hemozero_kg",
                "target_column": dataset["target_column"],
                "target_disease_id": target_disease["id"],
            }, folds)
            repository.upsert_cohort_associations(rows)
    except Exception as exc:
        return ToolResult("kg_build_fold_associations", Status.BLOCKED, f"Fold association graph failed: {exc}")
    output = context.run_dir / "knowledge_graph" / "fold_associations.jsonl"
    write_jsonl(output, rows)
    methods = pd.Series([row["method"] for row in rows]).value_counts().to_dict() if rows else {}
    return ToolResult("kg_build_fold_associations", Status.COMPLETE,
                      "Fold-local association subgraphs created without independent holdout labels",
                      [ArtifactRef(str(output), sha256_file(output), "application/x-ndjson")],
                      {"folds": len(folds), "associations": len(rows), "methods": methods,
                       "holdout_labels_accessed": False})
