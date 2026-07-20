from __future__ import annotations

import pandas as pd

from hemzero.common.hashing import sha256_file
from hemzero.common.io import atomic_json
from hemzero.common.schemas import ArtifactRef, Status, ToolResult
from hemzero.data.inspection import inspect_frame

from ._shared import ToolContext, require_file


def run(arguments: dict, context: ToolContext) -> ToolResult:
    dataset = context.config("dataset.yaml")
    train_path = require_file(context.project_root / arguments.get("train_path", dataset["train_path"]), "train_path")
    test_path = require_file(context.project_root / arguments.get("test_path", dataset["test_path"]), "test_path")
    train = pd.read_csv(train_path)
    test_features = pd.read_csv(test_path, usecols=lambda column: column != dataset["target_column"])
    required_features = set(dataset["raw_features"] + dataset["ratio_features"])
    missing = sorted(required_features - set(train) | required_features - set(test_features))
    if missing:
        raise ValueError(f"Missing configured features: {missing}")
    overlap = set(train[dataset["id_column"]]) & set(test_features[dataset["id_column"]])
    if overlap:
        raise ValueError(f"Train/test patient overlap: {sorted(overlap)[:5]}")
    fingerprint_columns = dataset["raw_features"] + dataset["ratio_features"]
    train_fingerprints = pd.util.hash_pandas_object(train[fingerprint_columns], index=False)
    test_fingerprints = pd.util.hash_pandas_object(test_features[fingerprint_columns], index=False)
    shared_fingerprints = set(train_fingerprints) & set(test_fingerprints)
    shared_test_rows = int(test_fingerprints.isin(shared_fingerprints).sum())
    report = {
        "train": inspect_frame(train, id_column=dataset["id_column"], target_column=dataset["target_column"]),
        "test": {"rows": len(test_features), "columns_without_target": len(test_features.columns),
                 "duplicate_ids": int(test_features[dataset["id_column"]].duplicated().sum())},
        "train_test_id_overlap": 0,
        "train_test_exact_feature_vector_overlap": shared_test_rows,
        "exact_feature_vector_overlap_authorized": bool(dataset.get("allow_exact_feature_vector_overlap", False)),
        "overlap_authorization": dataset.get("overlap_authorization"),
        "input_hashes": {"train": sha256_file(train_path), "test": sha256_file(test_path)},
        "test_target_accessed": False,
    }
    output = context.run_dir / "dataset" / "inspection.json"
    atomic_json(output, report)
    overlap_authorized = bool(dataset.get("allow_exact_feature_vector_overlap", False))
    status = Status.FAILED if shared_test_rows and not overlap_authorized else Status.COMPLETE
    summary = (
        "User-authorized exact cross-split feature overlap recorded for raw-data sensitivity run"
        if shared_test_rows and overlap_authorized
        else "Fatal cross-split exact feature-vector overlap detected"
        if shared_test_rows
        else "Dataset schema, patient separation and feature fingerprints validated"
    )
    return ToolResult("inspect_dataset", status, summary,
                      [ArtifactRef(str(output), sha256_file(output), "application/json")], report)
