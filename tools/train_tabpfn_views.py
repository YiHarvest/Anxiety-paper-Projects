from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from hemzero.common.hashing import sha256_file
from hemzero.common.schemas import ArtifactRef, Status, ToolResult
from hemzero.data.views import build_views
from hemzero.fusion.gating import data_quality_gate, stability_gate
from hemzero.models.tabpfn_views import repeated_fit_predict, tabpfn_capability

from ._shared import ToolContext, require_file


def _partition_indices(partition: pd.DataFrame, id_to_index: dict) -> tuple[np.ndarray, np.ndarray]:
    train_ids = partition[partition.split == "train"].patient_id
    validation_ids = partition[partition.split == "validation"].patient_id
    return (
        np.asarray([id_to_index[value] for value in train_ids]),
        np.asarray([id_to_index[value] for value in validation_ids]),
    )


def run(arguments: dict, context: ToolContext) -> ToolResult:
    load_dotenv(context.project_root / ".env", override=False)
    status = tabpfn_capability()
    if not status.available or not status.authenticated:
        reason = status.reason if not status.available else "TABPFN_TOKEN is not configured"
        return ToolResult("train_tabpfn_views", Status.BLOCKED, reason,
                          evidence={"backend": "tabpfn", "proxy_used": False})

    dataset = context.config("dataset.yaml")
    experiment = context.config("experiment.yaml")
    models = context.config("models.yaml")
    execution = experiment.get("execution", {})
    repeats = int(arguments.get("outer_repeats", execution.get("outer_repeats", experiment["outer_repeats"])))
    prediction_seeds = int(arguments.get("prediction_seeds", execution.get("tabpfn_prediction_seeds", 2)))
    device = str(models["tabpfn"].get("device", "cpu"))
    if len(pd.read_csv(context.run_dir / "dataset" / "cleaned_train.csv")) > int(models["tabpfn"]["max_cpu_rows_without_override"]):
        if not models["tabpfn"].get("allow_cpu_large_dataset", False):
            return ToolResult("train_tabpfn_views", Status.BLOCKED, "CPU row limit exceeded")
        os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "true"

    frame = pd.read_csv(require_file(context.run_dir / "dataset" / "cleaned_train.csv", "cleaned_train"))
    splits = pd.read_csv(require_file(context.run_dir / "splits" / "outer_split_assignment.csv", "outer_splits"))
    views = build_views(frame, dataset["raw_features"], dataset["ratio_features"])
    id_to_index = {patient_id: index for index, patient_id in enumerate(frame[dataset["id_column"]])}
    rows: list[dict] = []
    for repeat in sorted(splits.repeat.unique())[:repeats]:
        repeated = splits[splits.repeat == repeat]
        for fold in sorted(repeated.fold.unique()):
            partition = repeated[repeated.fold == fold]
            train_indices, valid_indices = _partition_indices(partition, id_to_index)
            fold_values: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            for view_offset, view_name in enumerate(("raw", "ratio", "full")):
                print(f"TabPFN repeat={repeat} fold={fold} view={view_name}", flush=True)
                probability, variance = repeated_fit_predict(
                    views[view_name].iloc[train_indices],
                    frame.iloc[train_indices][dataset["target_column"]].to_numpy(),
                    views[view_name].iloc[valid_indices],
                    device=device,
                    seed=int(experiment["seed"]) + 1000 * int(repeat) + 100 * int(fold) + 10 * view_offset,
                    prediction_seeds=prediction_seeds,
                )
                quality = data_quality_gate(
                    views[view_name].iloc[train_indices], views[view_name].iloc[valid_indices]
                )
                gate = quality * stability_gate(variance)
                fold_values[view_name] = probability, variance, gate
            for position, index in enumerate(valid_indices):
                row = {
                    "patient_id": frame.iloc[index][dataset["id_column"]],
                    "target": frame.iloc[index][dataset["target_column"]],
                    "repeat": int(repeat),
                    "fold": int(fold),
                }
                for view_name, (probability, variance, gate) in fold_values.items():
                    row[f"{view_name}_probability"] = float(probability[position])
                    row[f"{view_name}_variance"] = float(variance[position])
                    row[f"{view_name}_gate"] = float(gate[position])
                rows.append(row)

    oof_path = context.run_dir / "predictions" / "tabpfn_view_oof.csv"
    pd.DataFrame(rows).to_csv(oof_path, index=False)

    # Once the protocol is fixed, train on all development data and predict the untouched holdout.
    test = pd.read_csv(context.project_root / dataset["test_path"], usecols=lambda c: c != dataset["target_column"])
    test_views = build_views(test, dataset["raw_features"], dataset["ratio_features"])
    external = pd.DataFrame({"patient_id": test[dataset["id_column"]]})
    for view_offset, view_name in enumerate(("raw", "ratio", "full")):
        print(f"TabPFN final holdout view={view_name}", flush=True)
        probability, variance = repeated_fit_predict(
            views[view_name], frame[dataset["target_column"]].to_numpy(), test_views[view_name],
            device=device, seed=int(experiment["seed"]) + 9000 + 10 * view_offset,
            prediction_seeds=prediction_seeds,
        )
        quality = data_quality_gate(views[view_name], test_views[view_name])
        external[f"{view_name}_probability"] = probability
        external[f"{view_name}_variance"] = variance
        external[f"{view_name}_gate"] = quality * stability_gate(variance)
    external_path = context.run_dir / "predictions" / "tabpfn_view_holdout.csv"
    external.to_csv(external_path, index=False)

    evidence = {
        "backend": "TabPFN-3",
        "device": device,
        "proxy_used": False,
        "outer_repeats_designed": int(experiment["outer_repeats"]),
        "outer_repeats_executed": repeats,
        "folds": int(splits.fold.nunique()),
        "prediction_seeds": prediction_seeds,
        "oof_rows": len(rows),
        "holdout_rows": len(external),
        "test_labels_accessed": False,
        "execution_profile": execution.get("profile", "full"),
    }
    artifacts = [ArtifactRef(str(path), sha256_file(path), "text/csv") for path in (oof_path, external_path)]
    return ToolResult("train_tabpfn_views", Status.COMPLETE,
                      "Real multi-view TabPFN OOF and untouched holdout probabilities created", artifacts, evidence)
