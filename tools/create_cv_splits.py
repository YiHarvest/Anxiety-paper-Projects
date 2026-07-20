from __future__ import annotations

import pandas as pd

from hemzero.common.hashing import sha256_file
from hemzero.common.schemas import ArtifactRef, Status, ToolResult
from hemzero.validation.splits import create_repeated_splits

from ._shared import ToolContext, require_file


def run(arguments: dict, context: ToolContext) -> ToolResult:
    dataset, experiment = context.config("dataset.yaml"), context.config("experiment.yaml")
    train_path = require_file(context.run_dir / "dataset" / "cleaned_train.csv", "cleaned_train")
    frame = pd.read_csv(train_path)
    splits = create_repeated_splits(
        frame[dataset["id_column"]].to_numpy(), frame[dataset["target_column"]].to_numpy(),
        n_splits=int(experiment["outer_splits"]), n_repeats=int(experiment["outer_repeats"]), seed=int(experiment["seed"]),
    )
    output = context.run_dir / "splits" / "outer_split_assignment.csv"
    splits.to_csv(output, index=False)
    counts = splits.groupby(["repeat", "fold", "split"]).size()
    evidence = {"assignments": len(splits), "repeats": int(splits.repeat.nunique()), "folds": int(splits.fold.nunique()),
                "minimum_partition_size": int(counts.min())}
    return ToolResult("create_cv_splits", Status.COMPLETE, "Seed-locked repeated outer splits created",
                      [ArtifactRef(str(output), sha256_file(output), "text/csv")], evidence)
