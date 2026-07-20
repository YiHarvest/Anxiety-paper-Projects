from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from hemzero.common.hashing import sha256_file
from hemzero.common.schemas import ArtifactRef, Status, ToolResult
from hemzero.reporting.figures import plot_auroc
from hemzero.validation.bootstrap import bootstrap_intervals
from hemzero.validation.metrics import classification_metrics

from ._shared import ToolContext


def _oof_long(run_dir) -> pd.DataFrame:
    frames = []
    for filename in ("baseline_oof.csv", "ebm_oof.csv", "pysr_oof.csv"):
        path = run_dir / "predictions" / filename
        if path.exists():
            item = pd.read_csv(path)
            if "repeat" not in item:
                item["repeat"] = 0
            frames.append(item[["patient_id", "target", "repeat", "model", "probability"]])
    fused_path = run_dir / "predictions" / "hemozero_oof.csv"
    if fused_path.exists():
        frame = pd.read_csv(fused_path)
        mapping = {
            "TabPFN-Raw": "raw_probability", "TabPFN-Ratio": "ratio_probability",
            "TabPFN-Full": "full_probability", "ViewMean": "mean_probability",
            "ViewStacking": "stacking_probability", "PrecisionFusion": "precision_probability",
            "HemoZero": "hemozero_probability",
        }
        for model, column in mapping.items():
            frames.append(frame[["patient_id", "target", "repeat", column]].rename(columns={column: "probability"}).assign(model=model))
    return pd.concat(frames, ignore_index=True)


def _holdout_long(run_dir) -> pd.DataFrame:
    frames = []
    for filename in ("baseline_holdout.csv", "ebm_holdout.csv", "pysr_holdout.csv"):
        path = run_dir / "predictions" / filename
        if path.exists():
            frames.append(pd.read_csv(path)[["patient_id", "model", "probability"]])
    path = run_dir / "predictions" / "hemozero_holdout.csv"
    if path.exists():
        frame = pd.read_csv(path)
        mapping = {
            "TabPFN-Raw": "raw_probability", "TabPFN-Ratio": "ratio_probability",
            "TabPFN-Full": "full_probability", "ViewMean": "mean_probability",
            "ViewStacking": "stacking_probability", "PrecisionFusion": "precision_probability",
            "HemoZero": "hemozero_probability",
        }
        for model, column in mapping.items():
            frames.append(frame[["patient_id", column]].rename(columns={column: "probability"}).assign(model=model))
    return pd.concat(frames, ignore_index=True)


def run(arguments: dict, context: ToolContext) -> ToolResult:
    experiment, dataset = context.config("experiment.yaml"), context.config("dataset.yaml")
    raw_oof = _oof_long(context.run_dir)
    if raw_oof.empty:
        return ToolResult("evaluation", Status.BLOCKED, "No OOF prediction artifacts are available")
    repeat_rows = []
    for (model, repeat), group in raw_oof.groupby(["model", "repeat"]):
        repeat_rows.append({"model": model, "repeat": int(repeat), **classification_metrics(group.target, group.probability)})
    repeat_metrics = pd.DataFrame(repeat_rows)
    repeat_metrics_path = context.run_dir / "reports" / "repeat_metrics.csv"
    repeat_metrics.to_csv(repeat_metrics_path, index=False)
    oof = raw_oof.groupby(["patient_id", "model"], as_index=False).agg(target=("target", "first"), probability=("probability", "mean"))
    metrics_rows, bootstrap_rows = [], []
    for model, group in oof.groupby("model"):
        metrics_rows.append({"model": model, "evaluation": "development_oof", **classification_metrics(group.target, group.probability)})
        for row in bootstrap_intervals(group.target, group.probability, threshold=0.5,
                                       n_resamples=int(experiment["bootstrap_resamples"]), seed=int(experiment["seed"])):
            bootstrap_rows.append({"model": model, **row})
    metrics = pd.DataFrame(metrics_rows).sort_values("AUROC", ascending=False)
    metrics_path = context.run_dir / "reports" / "oof_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    bootstrap_path = context.run_dir / "reports" / "oof_bootstrap_intervals.csv"
    pd.DataFrame(bootstrap_rows).to_csv(bootstrap_path, index=False)

    holdout = _holdout_long(context.run_dir)
    # This is the first and only stage that reads independent holdout labels.
    labels = pd.read_csv(context.project_root / dataset["test_path"], usecols=[dataset["id_column"], dataset["target_column"]])
    holdout = holdout.merge(labels, left_on="patient_id", right_on=dataset["id_column"], validate="many_to_one")
    holdout_metrics = []
    for model, group in holdout.groupby("model"):
        holdout_metrics.append({"model": model, "evaluation": "independent_holdout",
                                **classification_metrics(group[dataset["target_column"]], group.probability)})
    holdout_metrics_path = context.run_dir / "reports" / "holdout_metrics.csv"
    pd.DataFrame(holdout_metrics).sort_values("AUROC", ascending=False).to_csv(holdout_metrics_path, index=False)

    fused_raw = pd.read_csv(context.run_dir / "predictions" / "hemozero_oof.csv")
    fused = fused_raw.groupby("patient_id", as_index=False).agg(
        target=("target", "first"), hemozero_probability=("hemozero_probability", "mean"),
        uncertainty_score=("uncertainty_score", "mean"), flip_rate=("flip_rate", "mean"),
    )
    abstention_rows = []
    for fraction in experiment["uncertainty"]["abstention_fractions"]:
        keep = int(np.ceil(len(fused) * (1 - float(fraction))))
        subset = fused.sort_values("uncertainty_score").iloc[:keep]
        values = classification_metrics(subset.target, subset.hemozero_probability)
        abstention_rows.append({"abstention_fraction": fraction, "retained": len(subset), **values})
    abstention_path = context.run_dir / "reports" / "selective_prediction.csv"
    pd.DataFrame(abstention_rows).to_csv(abstention_path, index=False)
    error = ((fused.hemozero_probability >= 0.5) != fused.target).astype(int)
    uncertainty_error_auc = float(roc_auc_score(error, fused.uncertainty_score)) if error.nunique() == 2 else float("nan")
    figure_path = context.run_dir / "figures" / "oof_auroc.png"
    plot_auroc(metrics, figure_path)
    paths = (metrics_path, repeat_metrics_path, bootstrap_path, holdout_metrics_path, abstention_path, figure_path)
    artifacts = [ArtifactRef(str(path), sha256_file(path)) for path in paths]
    return ToolResult("evaluation", Status.COMPLETE,
                      "OOF, bootstrap, calibration, uncertainty and independent holdout evaluation complete",
                      artifacts, {"models_evaluated": len(metrics), "bootstrap_resamples": experiment["bootstrap_resamples"],
                                  "patient_level_repeat_aggregation": True,
                                  "uncertainty_error_auc": uncertainty_error_auc,
                                  "test_labels_accessed": True, "test_labels_used_for_tuning": False})
