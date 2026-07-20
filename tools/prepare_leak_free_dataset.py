#!/usr/bin/env python3
"""Create an audited training set with train/holdout feature duplicates removed."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    np.random.seed(42)
    random.seed(42)

    root = args.project_root.resolve()
    config = yaml.safe_load((root / "configs" / "dataset.yaml").read_text(encoding="utf-8"))
    train_path = root / config["train_path"]
    test_path = root / config["test_path"]
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    features = list(config["raw_features"]) + list(config["ratio_features"])
    id_column = str(config["id_column"])

    train_hash = pd.util.hash_pandas_object(train[features], index=False)
    test_hash = pd.util.hash_pandas_object(test[features], index=False)
    remove = train_hash.isin(set(test_hash))
    cleaned = train.loc[~remove].copy()
    log = pd.DataFrame({
        "source_row_index": train.index[remove],
        "patient_id": train.loc[remove, id_column].to_numpy(),
        "action": "removed_from_training",
        "reason": "exact_15_feature_duplicate_in_final_holdout",
        "approved_by": "researcher_full_pipeline_authorization",
    })

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output / "anxiety_train_leak_free.csv", index=False)
    test.to_csv(output / "anxiety_test_unchanged.csv", index=False)
    log.to_csv(output / "cleaning_log.csv", index=False)

    after_hash = pd.util.hash_pandas_object(cleaned[features], index=False)
    overlap_after = int(test_hash.isin(set(after_hash)).sum())
    print(f"Training rows before: {len(train)}")
    print(f"Training rows removed: {int(remove.sum())}")
    print(f"Training rows after: {len(cleaned)}")
    print(f"Holdout rows unchanged: {len(test)}")
    print(f"Exact train-holdout feature overlap after cleaning: {overlap_after}")
    print("This code implements ONLY the cleaning rules you approved. Review the cleaning_log.csv output to verify all changes before proceeding to analysis.")
    if overlap_after != 0 or len(log) != 96:
        raise SystemExit("Leakage cleaning verification failed")


if __name__ == "__main__":
    main()
