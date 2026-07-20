from __future__ import annotations

import numpy as np
import pandas as pd

from hemzero.common.hashing import sha256_file
from hemzero.common.schemas import ArtifactRef, Status, ToolResult
from hemzero.models.baselines import baseline_factories

from ._shared import ToolContext, require_file


def run(arguments: dict, context: ToolContext) -> ToolResult:
    dataset, experiment = context.config("dataset.yaml"), context.config("experiment.yaml")
    execution = experiment.get("execution", {})
    frame = pd.read_csv(require_file(context.run_dir / "dataset" / "cleaned_train.csv", "cleaned_train"))
    splits = pd.read_csv(require_file(context.run_dir / "splits" / "outer_split_assignment.csv", "outer_splits"))
    id_to_index = {patient_id: index for index, patient_id in enumerate(frame[dataset["id_column"]])}
    base = baseline_factories(int(experiment["seed"]))
    specifications = {
        "Logistic-Raw": (base["logistic"], dataset["raw_features"]),
        "ElasticNet-Raw": (base["elastic_net"], dataset["raw_features"]),
        "ElasticNet-Full": (base["elastic_net"], dataset["raw_features"] + dataset["ratio_features"]),
        "SVM-Full": (base["svm"], dataset["raw_features"] + dataset["ratio_features"]),
        "RandomForest-Full": (base["random_forest"], dataset["raw_features"] + dataset["ratio_features"]),
    }
    repeats_to_run = int(arguments.get("outer_repeats", execution.get("outer_repeats", experiment["outer_repeats"])))
    repeats = sorted(splits.repeat.unique())[:repeats_to_run]
    rows = []
    for repeat in repeats:
        repeated = splits[splits.repeat == repeat]
        for fold in sorted(repeated.fold.unique()):
            train_ids = repeated[(repeated.fold == fold) & (repeated.split == "train")].patient_id
            valid_ids = repeated[(repeated.fold == fold) & (repeated.split == "validation")].patient_id
            train_indices = np.asarray([id_to_index[value] for value in train_ids])
            valid_indices = np.asarray([id_to_index[value] for value in valid_ids])
            for name, (factory, features) in specifications.items():
                model = factory()
                model.fit(frame.iloc[train_indices][features], frame.iloc[train_indices][dataset["target_column"]])
                probability = model.predict_proba(frame.iloc[valid_indices][features])[:, 1]
                rows.extend({
                    "patient_id": frame.iloc[index][dataset["id_column"]],
                    "target": frame.iloc[index][dataset["target_column"]], "repeat": int(repeat),
                    "fold": int(fold), "model": name, "probability": float(value),
                } for index, value in zip(valid_indices, probability, strict=True))
    oof_path = context.run_dir / "predictions" / "baseline_oof.csv"
    pd.DataFrame(rows).to_csv(oof_path, index=False)

    test = pd.read_csv(context.project_root / dataset["test_path"], usecols=lambda c: c != dataset["target_column"])
    holdout_rows = []
    for name, (factory, features) in specifications.items():
        model = factory().fit(frame[features], frame[dataset["target_column"]])
        probability = model.predict_proba(test[features])[:, 1]
        holdout_rows.extend({"patient_id": patient, "model": name, "probability": float(value)}
                            for patient, value in zip(test[dataset["id_column"]], probability, strict=True))
    holdout_path = context.run_dir / "predictions" / "baseline_holdout.csv"
    pd.DataFrame(holdout_rows).to_csv(holdout_path, index=False)
    artifacts = [ArtifactRef(str(path), sha256_file(path), "text/csv") for path in (oof_path, holdout_path)]
    return ToolResult("train_baselines", Status.COMPLETE, "Protocol-specified baseline OOF and holdout predictions created",
                      artifacts, {"models": list(specifications), "oof_rows": len(rows),
                                  "repeats_executed": repeats, "test_labels_accessed": False})
