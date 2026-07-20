from __future__ import annotations

import json

from hemzero.common.hashing import sha256_file
from hemzero.common.io import atomic_json
from hemzero.common.schemas import ArtifactRef, Status, ToolResult

from ._shared import ToolContext


def run(arguments: dict, context: ToolContext) -> ToolResult:
    required = [
        "dataset/inspection.json", "dataset/ratio_report.json", "dataset/feature_lineage_graph.json",
        "splits/outer_split_assignment.csv", "predictions/baseline_oof.csv",
        "predictions/tabpfn_view_oof.csv", "predictions/hemozero_oof.csv",
        "predictions/ebm_oof.csv", "predictions/pysr_oof.csv",
        "reports/oof_metrics.csv", "reports/holdout_metrics.csv", "reports/selective_prediction.csv",
        "evidence/tabdistill_interaction_selection.json", "formulas/pysr_formulas.json",
    ]
    missing = [name for name in required if not (context.run_dir / name).exists()]
    inspection = json.loads((context.run_dir / "dataset" / "inspection.json").read_text())
    ratio = json.loads((context.run_dir / "dataset" / "ratio_report.json").read_text())
    interaction_selection = json.loads((context.run_dir / "evidence" / "tabdistill_interaction_selection.json").read_text())
    evidence = {}
    for stage in ("train_baselines", "train_tabpfn_views", "hemzero_fusion", "tabdistill_ebm", "pysr_distillation", "evaluation"):
        path = context.run_dir / "evidence" / f"tool_{stage}.json"
        evidence[stage] = json.loads(path.read_text()) if path.exists() else {}
    tabpfn = evidence["train_tabpfn_views"].get("evidence", {})
    tabdistill = evidence["tabdistill_ebm"].get("evidence", {})
    pysr = evidence["pysr_distillation"].get("evidence", {})
    evaluation = evidence["evaluation"].get("evidence", {})
    checks = {
        "required_artifacts_present": not missing,
        "patient_overlap_zero": inspection.get("train_test_id_overlap") == 0,
        "feature_vector_overlap_zero": inspection.get("train_test_exact_feature_vector_overlap") == 0,
        "test_target_not_accessed_before_evaluation": all(
            evidence[name].get("evidence", {}).get("test_labels_accessed") is False
            for name in ("train_baselines", "train_tabpfn_views", "hemzero_fusion", "tabdistill_ebm", "pysr_distillation")
        ),
        "holdout_used_only_for_final_evaluation": evaluation.get("test_labels_accessed") is True and evaluation.get("test_labels_used_for_tuning") is False,
        "ratio_formulas_valid": ratio.get("invalid") == 0,
        "real_tabpfn_backend": tabpfn.get("backend") == "TabPFN-3" and tabpfn.get("proxy_used") is False,
        "fold_local_interaction_search": tabdistill.get("fold_local_search") is True,
        "selected_interactions_direction_stable": all(
            item.get("direction_consistency", 0) >= 0.6 for item in interaction_selection.get("selected", [])
        ),
        "real_pysr_backend": pysr.get("backend") == "PySR/SymbolicRegression.jl" and pysr.get("proxy_used") is False,
        "oof_soft_labels_required": pysr.get("outer_fold_search") is True,
    }
    report = {"checks": checks, "missing": missing, "pass": all(checks.values()) and not missing,
              "scope_limitations": {
                  "designed_outer_repeats": tabpfn.get("outer_repeats_designed"),
                  "executed_outer_repeats": tabpfn.get("outer_repeats_executed"),
                  "execution_profile": tabpfn.get("execution_profile"),
                  "tabdistill_search_repeats": tabdistill.get("search_repeats_executed"),
                  "tabdistill_folds_per_repeat": tabdistill.get("search_folds_per_repeat"),
              }}
    output = context.run_dir / "evidence" / "audit.json"
    atomic_json(output, report)
    return ToolResult("audit", Status.COMPLETE if report["pass"] else Status.FAILED,
                      "Evidence audit complete", [ArtifactRef(str(output), sha256_file(output))], report)
