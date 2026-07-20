from __future__ import annotations

import pandas as pd

from hemzero.common.hashing import sha256_file
from hemzero.common.io import atomic_json
from hemzero.common.schemas import ArtifactRef, Status, ToolResult
from hemzero.data.ratios import recompute_ratios

from ._shared import ToolContext, require_file


def run(arguments: dict, context: ToolContext) -> ToolResult:
    dataset, formulas = context.config("dataset.yaml"), context.config("ratio_formulas.yaml")
    train_path = require_file(context.project_root / arguments.get("train_path", dataset["train_path"]), "train_path")
    frame = pd.read_csv(train_path)
    cleaned, validation = recompute_ratios(frame, formulas)
    tolerance = float(context.config("audit_rules.yaml")["rules"]["ratio_relative_tolerance"])
    validation["valid"] = validation["relative_error"].fillna(float("inf")) <= tolerance
    validation_path = context.run_dir / "dataset" / "ratio_validation.csv"
    cleaned_path = context.run_dir / "dataset" / "cleaned_train.csv"
    report_path = context.run_dir / "dataset" / "ratio_report.json"
    validation.to_csv(validation_path, index=False)
    cleaned.to_csv(cleaned_path, index=False)
    report = {"rows": len(validation), "valid": int(validation.valid.sum()),
              "invalid": int((~validation.valid).sum()), "tolerance": tolerance,
              "near_zero_denominators": int(validation.denominator_near_zero.sum())}
    atomic_json(report_path, report)
    status = Status.COMPLETE if report["invalid"] == 0 else Status.FAILED
    return ToolResult("validate_ratios", status, "Ratio formula validation complete",
                      [ArtifactRef(str(path), sha256_file(path)) for path in (validation_path, cleaned_path, report_path)], report)
