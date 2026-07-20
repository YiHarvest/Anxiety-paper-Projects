from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from hemzero.common.hashing import sha256_file
from hemzero.common.io import atomic_json
from hemzero.common.schemas import ArtifactRef, Status, ToolResult
from hemzero.distillation.pysr_student import capability, train_pysr

from ._shared import ToolContext, require_file


def _select_features(frame: pd.DataFrame, candidates: list[str], target: np.ndarray, count: int) -> list[str]:
    scores = {}
    for name in candidates:
        correlation = np.corrcoef(np.asarray(frame[name], float), target)[0, 1]
        scores[name] = 0.0 if not np.isfinite(correlation) else abs(float(correlation))
    return sorted(candidates, key=lambda name: (-scores[name], name))[:count]


def _safe_probability(logits: np.ndarray, target_probability: np.ndarray) -> tuple[np.ndarray, int]:
    """Convert symbolic logits to finite probabilities without hiding replacement counts."""

    values = np.asarray(logits, dtype=float)
    invalid = ~np.isfinite(values)
    fallback_probability = float(np.clip(np.mean(target_probability), 1e-6, 1 - 1e-6))
    fallback_logit = float(logit(fallback_probability))
    values = np.nan_to_num(values, nan=fallback_logit, posinf=30.0, neginf=-30.0)
    return expit(np.clip(values, -30.0, 30.0)), int(invalid.sum())


def run(arguments: dict, context: ToolContext) -> ToolResult:
    status = capability()
    if not status.available:
        return ToolResult("pysr_distillation", Status.BLOCKED, status.reason)
    dataset, experiment = context.config("dataset.yaml"), context.config("experiment.yaml")
    models, execution = context.config("models.yaml"), experiment.get("execution", {})
    niterations = int(arguments.get("niterations", execution.get("pysr_niterations", models["pysr"]["niterations"])))
    repeat_count = int(execution.get("outer_repeats", experiment["outer_repeats"]))
    feature_count = int(models["pysr"]["max_input_features"])
    base = pd.read_csv(require_file(context.run_dir / "dataset" / "cleaned_train.csv", "cleaned_train"))
    teacher = pd.read_csv(require_file(context.run_dir / "predictions" / "hemozero_oof.csv", "hemozero_oof"))
    splits = pd.read_csv(require_file(context.run_dir / "splits" / "outer_split_assignment.csv", "outer_splits"))
    id_to_index = {value: i for i, value in enumerate(base[dataset["id_column"]])}
    rows, formula_records = [], []
    nonfinite_predictions_replaced = 0
    formula_root = context.run_dir / "formulas" / "pysr_search"
    for repeat in sorted(splits.repeat.unique())[:repeat_count]:
        repeated_teacher = teacher[teacher.repeat == repeat].set_index("patient_id")
        frame = base.copy()
        frame["teacher_probability"] = frame[dataset["id_column"]].map(repeated_teacher["hemozero_probability"])
        repeated_splits = splits[splits.repeat == repeat]
        for fold in sorted(repeated_splits.fold.unique()):
            partition = repeated_splits[repeated_splits.fold == fold]
            train_idx = np.asarray([id_to_index[v] for v in partition[partition.split == "train"].patient_id])
            valid_idx = np.asarray([id_to_index[v] for v in partition[partition.split == "validation"].patient_id])
            selected = _select_features(frame.iloc[train_idx], dataset["raw_features"],
                                        frame.iloc[train_idx]["teacher_probability"].to_numpy(), feature_count)
            for mode in ("Direct", "Distilled"):
                target_probability = (np.where(frame.iloc[train_idx][dataset["target_column"]].to_numpy() == 1, 0.95, 0.05)
                                      if mode == "Direct" else frame.iloc[train_idx]["teacher_probability"].to_numpy())
                print(f"PySR repeat={repeat} fold={fold} mode={mode} iterations={niterations}", flush=True)
                model = train_pysr(
                    frame.iloc[train_idx][selected], target_probability, niterations=niterations,
                    output_directory=str(formula_root), run_id=f"r{repeat}-f{fold}-{mode.lower()}",
                    seed=int(experiment["seed"]) + 100 * int(repeat) + int(fold),
                    maxsize=int(models["pysr"]["maxsize"]), maxdepth=int(models["pysr"]["maxdepth"]),
                )
                probability, replaced = _safe_probability(
                    model.predict(frame.iloc[valid_idx][selected]), target_probability
                )
                nonfinite_predictions_replaced += replaced
                formula_records.append({"repeat": int(repeat), "fold": int(fold), "mode": mode,
                                        "features": selected, "formula": str(model.sympy()),
                                        "nonfinite_predictions_replaced": replaced})
                rows.extend({"patient_id": frame.iloc[index][dataset["id_column"]],
                             "target": frame.iloc[index][dataset["target_column"]],
                             "repeat": int(repeat), "fold": int(fold), "model": f"PySR-{mode}",
                             "probability": float(value)}
                            for index, value in zip(valid_idx, probability, strict=True))
    oof_path = context.run_dir / "predictions" / "pysr_oof.csv"
    pd.DataFrame(rows).to_csv(oof_path, index=False)

    mean_teacher = teacher.groupby("patient_id", as_index=True).hemozero_probability.mean()
    final_frame = base.copy()
    final_frame["teacher_probability"] = final_frame[dataset["id_column"]].map(mean_teacher)
    final_selected = _select_features(final_frame, dataset["raw_features"],
                                      final_frame["teacher_probability"].to_numpy(), feature_count)
    test = pd.read_csv(context.project_root / dataset["test_path"], usecols=lambda c: c != dataset["target_column"])
    holdout_rows, final_formulas = [], []
    for mode in ("Direct", "Distilled"):
        target_probability = (np.where(final_frame[dataset["target_column"]].to_numpy() == 1, 0.95, 0.05)
                              if mode == "Direct" else final_frame["teacher_probability"].to_numpy())
        model = train_pysr(
            final_frame[final_selected], target_probability, niterations=niterations,
            output_directory=str(formula_root), run_id=f"final-{mode.lower()}", seed=int(experiment["seed"]),
            maxsize=int(models["pysr"]["maxsize"]), maxdepth=int(models["pysr"]["maxdepth"]),
        )
        probability, replaced = _safe_probability(model.predict(test[final_selected]), target_probability)
        nonfinite_predictions_replaced += replaced
        final_formulas.append({"mode": mode, "features": final_selected, "formula": str(model.sympy()),
                               "nonfinite_predictions_replaced": replaced})
        holdout_rows.extend({"patient_id": patient, "model": f"PySR-{mode}", "probability": float(value)}
                            for patient, value in zip(test[dataset["id_column"]], probability, strict=True))
    holdout_path = context.run_dir / "predictions" / "pysr_holdout.csv"
    pd.DataFrame(holdout_rows).to_csv(holdout_path, index=False)
    formulas_path = context.run_dir / "formulas" / "pysr_formulas.json"
    atomic_json(formulas_path, {"outer_fold_formulas": formula_records, "final_formulas": final_formulas})
    artifacts = [ArtifactRef(str(path), sha256_file(path)) for path in (oof_path, holdout_path, formulas_path)]
    evidence = {"backend": "PySR/SymbolicRegression.jl", "proxy_used": False, "niterations": niterations,
                "outer_fold_search": True, "outer_repeats_executed": repeat_count,
                "modes": ["Direct", "Distilled"], "input_scope": "raw biomarkers only",
                "nonfinite_predictions_replaced": nonfinite_predictions_replaced,
                "test_labels_accessed": False}
    return ToolResult("pysr_distillation", Status.COMPLETE,
                      "Repeated outer-fold and final PySR direct/distilled formulas created", artifacts, evidence)
