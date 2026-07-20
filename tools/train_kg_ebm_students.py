from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import joblib
import numpy as np
import pandas as pd

from hemzero.common.hashing import sha256_file
from hemzero.common.schemas import ArtifactRef, Status, ToolResult
from hemzero.distillation.ebm_student import create_ebm
from hemzero.distillation.knowledge_constrained_distillation import ARM_MODEL_NAMES, build_ablation_arms
from hemzero.knowledge_graph.snapshot import read_jsonl

from ._kg_shared import open_repository
from ._shared import ToolContext, require_file


def _indices(rows: list[dict], features: list[str]) -> list[tuple[int, int]]:
    return [tuple(features.index(name) for name in row["features"]) for row in rows]


def _arm_quality(selections: dict[tuple[int, int], dict[str, list[dict]]], decisions: list[dict]) -> pd.DataFrame:
    decision_by_summary = {row["summary_id"]: row for row in decisions}
    rows = []
    for arm in ARM_MODEL_NAMES:
        fold_sets, selected_rows = [], []
        for arms in selections.values():
            chosen = arms[arm]
            fold_sets.append({row["pair_id"] for row in chosen})
            selected_rows.extend(chosen)
        jaccards = []
        for left, right in combinations(fold_sets, 2):
            jaccards.append(len(left & right) / len(left | right) if left | right else 1.0)
        assessments = [decision_by_summary[row["summary_id"]]["assessment"] for row in selected_rows]
        count = len(selected_rows)
        rows.append({
            "arm": arm, "model": ARM_MODEL_NAMES[arm], "selected_across_folds": count,
            "mechanical_interaction_retention_rate": (sum(a["direct_derivation"] or a["shares_component"] for a in assessments) / count) if count else 0.0,
            "direct_derivation_retention_rate": (sum(a["direct_derivation"] for a in assessments) / count) if count else 0.0,
            "shared_component_retention_rate": (sum(a["shares_component"] for a in assessments) / count) if count else 0.0,
            "mean_direction_consistency": float(np.mean([r["direction_consistency"] for r in selected_rows])) if count else np.nan,
            "mean_selection_frequency": float(np.mean([r["frequency"] for r in selected_rows])) if count else np.nan,
            "repeated_validation_jaccard": float(np.mean(jaccards)) if jaccards else np.nan,
            "supporting_evidence_coverage": (sum(a["has_supporting_evidence"] for a in assessments) / count) if count else 0.0,
            "conflicting_evidence_fraction": (sum(a["has_conflicting_evidence"] for a in assessments) / count) if count else 0.0,
            "stable_without_evidence_fraction": (sum(not a["has_supporting_evidence"] and not a["has_conflicting_evidence"] for a in assessments) / count) if count else 0.0,
        })
    return pd.DataFrame(rows)


def run(arguments: dict, context: ToolContext) -> ToolResult:
    dataset, experiment, models = context.config("dataset.yaml"), context.config("experiment.yaml"), context.config("models.yaml")
    policy = context.config("interaction_policy.yaml")
    frame = pd.read_csv(require_file(context.run_dir / "dataset" / "cleaned_train.csv", "cleaned_train"))
    splits = pd.read_csv(require_file(context.run_dir / "splits" / "outer_split_assignment.csv", "outer_splits"))
    summaries = read_jsonl(require_file(context.run_dir / "knowledge_graph" / "interaction_summaries.jsonl", "interaction_summaries"))
    decisions = read_jsonl(require_file(context.run_dir / "knowledge_graph" / "interaction_decisions.jsonl", "interaction_decisions"))
    features = dataset["raw_features"] + dataset["ratio_features"]
    id_to_index = {value: i for i, value in enumerate(frame[dataset["id_column"]])}
    decision_by_scope: dict[tuple[int | None, int | None], list[dict]] = defaultdict(list)
    summary_by_scope: dict[tuple[int | None, int | None], list[dict]] = defaultdict(list)
    for row in decisions:
        decision_by_scope[(row.get("repeat"), row.get("fold"))].append(row)
    for row in summaries:
        summary_by_scope[(row.get("repeat"), row.get("fold"))].append(row)
    selections: dict[tuple[int, int], dict[str, list[dict]]] = {}
    prediction_rows: list[dict] = []
    for (repeat, fold), partition in splits.groupby(["repeat", "fold"], sort=True):
        key = (int(repeat), int(fold))
        arms = build_ablation_arms(
            summary_by_scope[key], decision_by_scope[key], min_frequency=float(policy["min_frequency"]),
            min_direction_consistency=float(policy["min_direction_consistency"]),
            max_interactions=int(policy["max_interactions"]),
        )
        selections[key] = arms
        train_idx = np.asarray([id_to_index[value] for value in partition[partition.split == "train"].patient_id])
        valid_idx = np.asarray([id_to_index[value] for value in partition[partition.split == "validation"].patient_id])
        for arm, selected in arms.items():
            model = create_ebm(
                interactions=_indices(selected, features),
                seed=int(experiment["seed"]) + 100 * int(repeat) + int(fold),
                max_rounds=int(models["ebm"]["max_rounds"]), outer_bags=int(models["ebm"]["outer_bags"]),
            )
            model.fit(frame.iloc[train_idx][features], frame.iloc[train_idx][dataset["target_column"]])
            probability = model.predict_proba(frame.iloc[valid_idx][features])[:, 1]
            prediction_rows.extend({
                "patient_id": frame.iloc[index][dataset["id_column"]], "target": frame.iloc[index][dataset["target_column"]],
                "repeat": int(repeat), "fold": int(fold), "model": ARM_MODEL_NAMES[arm], "arm": arm,
                "probability": float(value), "selected_interactions": len(selected),
            } for index, value in zip(valid_idx, probability, strict=True))
    oof = pd.DataFrame(prediction_rows)
    oof_path = context.run_dir / "predictions" / "ebm_oof.csv"
    oof.to_csv(oof_path, index=False)

    global_key = (None, None)
    final_arms = build_ablation_arms(
        summary_by_scope[global_key], decision_by_scope[global_key], min_frequency=float(policy["min_frequency"]),
        min_direction_consistency=float(policy["min_direction_consistency"]),
        max_interactions=int(policy["max_interactions"]),
    )
    try:
        with open_repository(context) as repository:
            repository.upsert_ebm_models(context.run_dir.name, final_arms, ARM_MODEL_NAMES)
    except Exception as exc:
        return ToolResult("train_kg_ebm_students", Status.BLOCKED, f"EBM model provenance persistence failed: {exc}")
    test = pd.read_csv(context.project_root / dataset["test_path"], usecols=lambda column: column != dataset["target_column"])
    holdout_rows: list[dict] = []
    for arm, selected in final_arms.items():
        model = create_ebm(
            interactions=_indices(selected, features), seed=int(experiment["seed"]),
            max_rounds=int(models["ebm"]["max_rounds"]), outer_bags=int(models["ebm"]["outer_bags"]),
        )
        model.fit(frame[features], frame[dataset["target_column"]])
        probability = model.predict_proba(test[features])[:, 1]
        name = ARM_MODEL_NAMES[arm]
        holdout_rows.extend({"patient_id": patient, "model": name, "arm": arm, "probability": float(value),
                             "selected_interactions": len(selected)}
                            for patient, value in zip(test[dataset["id_column"]], probability, strict=True))
        joblib.dump(model, context.run_dir / "models" / f"{name.lower().replace('-', '_')}.joblib", compress=3)
    holdout_path = context.run_dir / "predictions" / "ebm_holdout.csv"
    pd.DataFrame(holdout_rows).to_csv(holdout_path, index=False)

    teacher = pd.read_csv(require_file(context.run_dir / "predictions" / "hemozero_oof.csv", "hemozero_oof"))
    merged = oof.merge(teacher[["patient_id", "repeat", "fold", "hemozero_probability"]],
                       on=["patient_id", "repeat", "fold"], validate="many_to_one")
    fidelity = []
    for (arm, model_name), group in merged.groupby(["arm", "model"]):
        correlation = group[["probability", "hemozero_probability"]].corr().iloc[0, 1]
        fidelity.append({
            "arm": arm, "model": model_name, "teacher_student_probability_correlation": float(correlation),
            "teacher_probability_mae": float(np.mean(np.abs(group.probability - group.hemozero_probability))),
            "teacher_student_classification_agreement": float(np.mean((group.probability >= 0.5) == (group.hemozero_probability >= 0.5))),
        })
    fidelity_path = context.run_dir / "reports" / "distillation_metrics.csv"
    pd.DataFrame(fidelity).to_csv(fidelity_path, index=False)
    quality_path = context.run_dir / "reports" / "interaction_quality_metrics.csv"
    _arm_quality(selections, decisions).to_csv(quality_path, index=False)
    paths = (oof_path, holdout_path, fidelity_path, quality_path)
    return ToolResult("train_kg_ebm_students", Status.COMPLETE,
                      "M0, M1, M2, M3-C and M3-D EBM students trained with fold-local decisions",
                      [ArtifactRef(str(path), sha256_file(path)) for path in paths],
                      {"models": list(ARM_MODEL_NAMES.values()), "oof_rows": len(oof),
                       "holdout_rows": len(holdout_rows), "final_interactions": {arm: len(rows) for arm, rows in final_arms.items()},
                       "test_labels_accessed": False})
