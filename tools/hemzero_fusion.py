from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from hemzero.common.hashing import sha256_file
from hemzero.common.schemas import ArtifactRef, Status, ToolResult
from hemzero.fusion.precision_fusion import precision_fusion
from hemzero.fusion.uncertainty import uncertainty_summary

from ._shared import ToolContext


PROBABILITY_COLUMNS = ["raw_probability", "ratio_probability", "full_probability"]
VARIANCE_COLUMNS = ["raw_variance", "ratio_variance", "full_variance"]
GATE_COLUMNS = ["raw_gate", "ratio_gate", "full_gate"]


def _add_fusions(frame: pd.DataFrame, *, stacker: LogisticRegression | None = None) -> tuple[pd.DataFrame, LogisticRegression]:
    probabilities = frame[PROBABILITY_COLUMNS].to_numpy(float)
    variances = frame[VARIANCE_COLUMNS].to_numpy(float)
    gates = frame[GATE_COLUMNS].to_numpy(float)
    gated, weights = precision_fusion(probabilities, variances, gates)
    precision, _ = precision_fusion(probabilities, variances)
    result = frame.copy()
    result["mean_probability"] = probabilities.mean(axis=1)
    result["precision_probability"] = precision
    result["hemozero_probability"] = gated
    if stacker is None:
        stacker = LogisticRegression(random_state=284, max_iter=2000).fit(probabilities, frame["target"])
    result["stacking_probability"] = stacker.predict_proba(probabilities)[:, 1]
    for name, values in uncertainty_summary(probabilities).items():
        result[name] = values
    # Include instability estimated from seed-level variance in the final uncertainty score.
    result["prediction_std"] = np.sqrt(variances.mean(axis=1))
    result["uncertainty_score"] = np.maximum(result["view_disagreement"], result["prediction_std"])
    result["view_agreement"] = ((probabilities >= 0.5) == (probabilities[:, [0]] >= 0.5)).mean(axis=1)
    result["abstain_recommended"] = result["uncertainty_score"] >= result["uncertainty_score"].quantile(0.9)
    for index, name in enumerate(("raw", "ratio", "full")):
        result[f"{name}_weight"] = weights[:, index]
    return result, stacker


def run(arguments: dict, context: ToolContext) -> ToolResult:
    source = context.run_dir / "predictions" / "tabpfn_view_oof.csv"
    holdout_source = context.run_dir / "predictions" / "tabpfn_view_holdout.csv"
    if not source.exists() or not holdout_source.exists():
        return ToolResult("hemzero_fusion", Status.BLOCKED, "TabPFN OOF/holdout view predictions are missing")
    frame = pd.read_csv(source)
    missing = sorted(set(PROBABILITY_COLUMNS + VARIANCE_COLUMNS + GATE_COLUMNS) - set(frame))
    if missing:
        raise ValueError(f"Missing fusion columns: {missing}")

    # Cross-fitted stacking predictions: each validation fold is scored by a meta-model
    # trained on other folds' OOF predictions from the same repeat.
    pieces = []
    for repeat in sorted(frame.repeat.unique()):
        repeated = frame[frame.repeat == repeat].copy()
        for fold in sorted(repeated.fold.unique()):
            validation = repeated[repeated.fold == fold].copy()
            meta_train = repeated[repeated.fold != fold]
            meta = LogisticRegression(random_state=284, max_iter=2000).fit(
                meta_train[PROBABILITY_COLUMNS], meta_train["target"]
            )
            fused, _ = _add_fusions(validation, stacker=meta)
            pieces.append(fused)
    output = pd.concat(pieces, ignore_index=True).sort_values(["repeat", "fold", "patient_id"])
    repeat_std = output.groupby("patient_id")["hemozero_probability"].transform("std").fillna(0.0)
    positive_fraction = output.groupby("patient_id")["hemozero_probability"].transform(lambda x: (x >= 0.5).mean())
    seed_std = np.sqrt(output[VARIANCE_COLUMNS].mean(axis=1))
    output["prediction_std"] = np.maximum(seed_std, repeat_std)
    output["flip_rate"] = np.minimum(positive_fraction, 1 - positive_fraction)
    output["uncertainty_score"] = np.maximum(output["view_disagreement"], output["prediction_std"])
    threshold = output.groupby("patient_id").uncertainty_score.mean().quantile(0.9)
    output["abstain_recommended"] = output["uncertainty_score"] >= threshold
    oof_path = context.run_dir / "predictions" / "hemozero_oof.csv"
    output.to_csv(oof_path, index=False)

    holdout = pd.read_csv(holdout_source)
    meta_frame = frame.groupby("patient_id", as_index=False).agg(
        {**{column: "mean" for column in PROBABILITY_COLUMNS}, "target": "first"}
    )
    final_stacker = LogisticRegression(random_state=284, max_iter=2000).fit(
        meta_frame[PROBABILITY_COLUMNS], meta_frame["target"]
    )
    external, _ = _add_fusions(holdout.assign(target=0), stacker=final_stacker)
    external = external.drop(columns="target")
    holdout_path = context.run_dir / "predictions" / "hemozero_holdout.csv"
    external.to_csv(holdout_path, index=False)

    artifacts = [ArtifactRef(str(path), sha256_file(path), "text/csv") for path in (oof_path, holdout_path)]
    evidence = {
        "oof_rows": len(output), "holdout_rows": len(external), "stacking_cross_fitted": True,
        "fusion_comparators": ["mean", "stacking", "precision", "precision_gated"],
        "test_labels_accessed": False,
    }
    return ToolResult("hemzero_fusion", Status.COMPLETE,
                      "Cross-fitted fusion comparators and HemoZero predictions created", artifacts, evidence)
