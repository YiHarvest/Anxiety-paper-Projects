from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from hemzero.common.hashing import sha256_file
from hemzero.common.io import atomic_json
from hemzero.common.schemas import ArtifactRef, Status, ToolResult
from hemzero.distillation.ebm_student import train_ebm_students
from hemzero.distillation.tabdistill_runner import capability, extract_interactions_from_data
from hemzero.graph.interaction_filter import filter_interactions
from hemzero.graph.lineage import build_lineage_graph

from ._shared import ToolContext, require_file


def _collapse_indices(payload: dict) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in payload.get("candidates", []):
        grouped[tuple(sorted(item["features"]))].append(item)
    collapsed = []
    for pair, items in grouped.items():
        best = max(items, key=lambda row: abs(float(row.get("score", 0))))
        collapsed.append({
            "features": list(pair),
            "frequency": max(float(row.get("frequency", 0)) for row in items),
            "score": float(best["score"]),
            "direction_consistency": float(best.get("direction_consistency", 0)),
            "supporting_indices": sorted({str(row.get("index_type")) for row in items}),
        })
    return collapsed


def _aggregate(payloads: list[dict]) -> list[dict]:
    values: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for payload in payloads:
        for item in _collapse_indices(payload):
            values[tuple(item["features"])].append(item)
    return [{
        "features": list(pair),
        "frequency": len(items) / len(payloads),
        "score": float(np.mean([row["score"] for row in items])),
        "direction_consistency": float(abs(np.mean(np.sign([row["score"] for row in items])))),
        "supporting_indices": sorted({index for row in items for index in row["supporting_indices"]}),
    } for pair, items in values.items()]


def run(arguments: dict, context: ToolContext) -> ToolResult:
    dataset, experiment = context.config("dataset.yaml"), context.config("experiment.yaml")
    models, execution = context.config("models.yaml"), experiment.get("execution", {})
    repository = Path(arguments.get("repository", models["tabdistill"]["repository"]))
    status = capability(repository)
    if not status.available:
        return ToolResult("tabdistill_ebm", Status.BLOCKED, status.reason)
    frame = pd.read_csv(require_file(context.run_dir / "dataset" / "cleaned_train.csv", "cleaned_train"))
    splits = pd.read_csv(require_file(context.run_dir / "splits" / "outer_split_assignment.csv", "outer_splits"))
    features = dataset["raw_features"] + dataset["ratio_features"]
    id_to_index = {value: i for i, value in enumerate(frame[dataset["id_column"]])}
    search_repeats = int(arguments.get("search_repeats", execution.get("tabdistill_repeats", 1)))
    folds_per_repeat = int(arguments.get("folds_per_repeat", execution.get("tabdistill_folds_per_repeat", 5)))
    execute_repeats = int(execution.get("outer_repeats", experiment["outer_repeats"]))
    sample_budget = int(arguments.get("sample_budget", models["tabdistill"]["sample_budget"]))
    max_samples = int(arguments.get("max_samples", execution.get("tabdistill_max_samples", 3)))
    index_types = list(models["tabdistill"]["index_types"])
    graph = build_lineage_graph(dataset["raw_features"], context.config("ratio_formulas.yaml"))
    payloads, local_selected = [], {}
    for repeat in sorted(splits.repeat.unique())[:search_repeats]:
        repeated = splits[splits.repeat == repeat]
        for fold in sorted(repeated.fold.unique())[:folds_per_repeat]:
            partition = repeated[repeated.fold == fold]
            train_ids = partition[partition.split == "train"].patient_id
            indices = [id_to_index[value] for value in train_ids]
            training_path = context.run_dir / "dataset" / f"tabdistill_outer_train_r{repeat}_f{fold}.csv"
            frame.iloc[indices][features + [dataset["target_column"]]].to_csv(training_path, index=False)
            output_path = context.run_dir / "evidence" / f"tabdistill_candidates_r{repeat}_f{fold}.json"
            print(f"TabDistill repeat={repeat} fold={fold} samples={max_samples} indices={len(index_types)}", flush=True)
            payload = extract_interactions_from_data(
                repository=repository, training_csv=training_path, output_json=output_path,
                feature_names=features, target_column=dataset["target_column"],
                sample_budget=sample_budget, max_samples=max_samples, index_types=index_types,
                seed=int(experiment["seed"]) + 100 * int(repeat) + int(fold),
            )
            payloads.append(payload)
            fold_selected, _ = filter_interactions(
                _collapse_indices(payload), graph, min_frequency=1 / max(max_samples, 1),
                max_interactions=int(models["ebm"]["max_interactions"]),
            )
            local_selected[(int(repeat), int(fold))] = fold_selected
    candidates = _aggregate(payloads)
    selected, rejected = filter_interactions(
        candidates, graph, min_frequency=0.4,
        max_interactions=int(models["ebm"]["max_interactions"]),
    )
    selection_path = context.run_dir / "evidence" / "tabdistill_interaction_selection.json"
    atomic_json(selection_path, {
        "candidates": candidates, "selected": selected, "rejected": rejected,
        "fold_local_selected": {f"r{r}_f{f}": rows for (r, f), rows in local_selected.items()},
    })

    rows = []
    for repeat in sorted(splits.repeat.unique())[:execute_repeats]:
        repeated = splits[splits.repeat == repeat]
        for fold in sorted(repeated.fold.unique()):
            partition = repeated[repeated.fold == fold]
            train_idx = np.asarray([id_to_index[v] for v in partition[partition.split == "train"].patient_id])
            valid_idx = np.asarray([id_to_index[v] for v in partition[partition.split == "validation"].patient_id])
            fold_selection = local_selected.get((int(repeat), int(fold)), selected)
            selected_indices = [tuple(features.index(name) for name in item["features"]) for item in fold_selection]
            students = train_ebm_students(
                frame.iloc[train_idx][features], frame.iloc[train_idx][dataset["target_column"]],
                selected_indices, seed=int(experiment["seed"]) + 100 * int(repeat) + int(fold),
                max_rounds=int(models["ebm"]["max_rounds"]), outer_bags=int(models["ebm"]["outer_bags"]),
            )
            for name, model in students.items():
                probability = model.predict_proba(frame.iloc[valid_idx][features])[:, 1]
                rows.extend({
                    "patient_id": frame.iloc[index][dataset["id_column"]],
                    "target": frame.iloc[index][dataset["target_column"]],
                    "repeat": int(repeat), "fold": int(fold), "model": name, "probability": float(value),
                } for index, value in zip(valid_idx, probability, strict=True))
    oof_path = context.run_dir / "predictions" / "ebm_oof.csv"
    pd.DataFrame(rows).to_csv(oof_path, index=False)

    global_indices = [tuple(features.index(name) for name in item["features"]) for item in selected]
    final_models = train_ebm_students(
        frame[features], frame[dataset["target_column"]], global_indices, seed=int(experiment["seed"]),
        max_rounds=int(models["ebm"]["max_rounds"]), outer_bags=int(models["ebm"]["outer_bags"]),
    )
    test = pd.read_csv(context.project_root / dataset["test_path"], usecols=lambda c: c != dataset["target_column"])
    holdout_rows = []
    for name, model in final_models.items():
        probability = model.predict_proba(test[features])[:, 1]
        holdout_rows.extend({"patient_id": patient, "model": name, "probability": float(value)}
                            for patient, value in zip(test[dataset["id_column"]], probability, strict=True))
        joblib.dump(model, context.run_dir / "models" / f"{name.lower().replace('-', '_')}.joblib", compress=3)
    holdout_path = context.run_dir / "predictions" / "ebm_holdout.csv"
    pd.DataFrame(holdout_rows).to_csv(holdout_path, index=False)
    artifacts = [ArtifactRef(str(path), sha256_file(path)) for path in (selection_path, oof_path, holdout_path)]
    evidence = {"backend": "Clouddelta/tab-distill", "teacher_backend": "TabPFN-3",
                "fold_local_search": True, "search_repeats_executed": search_repeats,
                "search_folds_per_repeat": folds_per_repeat, "interaction_indices": index_types,
                "candidate_payloads": len(payloads), "candidates": len(candidates), "selected": selected,
                "oof_repeats": execute_repeats, "oof_rows": len(rows), "test_labels_accessed": False}
    return ToolResult("tabdistill_ebm", Status.COMPLETE,
                      "Repeated fold-local multi-index TabDistill and cross-fitted EBM students complete",
                      artifacts, evidence)
