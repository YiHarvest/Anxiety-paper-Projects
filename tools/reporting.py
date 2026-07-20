from __future__ import annotations

import json

import pandas as pd

from hemzero.common.hashing import sha256_file
from hemzero.common.schemas import ArtifactRef, Status, ToolResult

from ._shared import ToolContext


def run(arguments: dict, context: ToolContext) -> ToolResult:
    audit_path = context.run_dir / "evidence" / "audit.json"
    if not audit_path.exists() or not json.loads(audit_path.read_text()).get("pass"):
        return ToolResult("reporting", Status.BLOCKED, "Passing audit artifact is required")
    audit = json.loads(audit_path.read_text())
    oof = pd.read_csv(context.run_dir / "reports" / "oof_metrics.csv")
    holdout = pd.read_csv(context.run_dir / "reports" / "holdout_metrics.csv")
    selective = pd.read_csv(context.run_dir / "reports" / "selective_prediction.csv")
    interactions = json.loads((context.run_dir / "evidence" / "tabdistill_interaction_selection.json").read_text())
    formulas = json.loads((context.run_dir / "formulas" / "pysr_formulas.json").read_text())
    distillation_path = context.run_dir / "reports" / "distillation_metrics.csv"
    interaction_quality_path = context.run_dir / "reports" / "interaction_quality_metrics.csv"
    inspection = json.loads((context.run_dir / "dataset" / "inspection.json").read_text())
    development_rows = int(inspection["train"]["rows"])
    holdout_rows = int(inspection["test"]["rows"])
    limitations = audit["scope_limitations"]
    def table(frame: pd.DataFrame) -> str:
        return "```csv\n" + frame.to_csv(index=False) + "```"

    full_primary = limitations["executed_outer_repeats"] == limitations["designed_outer_repeats"]
    limitation_text = (
        "This is the configured 5-repeat CPU primary computational run. It supports model-comparison "
        "claims for the supplied development and fixed holdout cohorts, but no causal or clinical-"
        "deployment claim. Independent-site and prospective validation remain required."
        if full_primary else
        "This is a reduced CPU end-to-end validation run. It proves backend identity, leakage controls "
        "and artifact contracts, but it does not replace the configured 5-repeat primary analysis."
    )
    report = f"""# HemoZero experiment report: {context.run_dir.name}

## Protocol status

- Real teacher backend: TabPFN-3 on CPU.
- Development cohort: {development_rows}; independent holdout: {holdout_rows}.
- Designed validation: {limitations['designed_outer_repeats']} × 5 outer folds.
- Executed profile: {limitations['execution_profile']} ({limitations['executed_outer_repeats']} × 5 outer folds).
- TabDistill interaction search: {limitations['tabdistill_search_repeats']} repeats × {limitations['tabdistill_folds_per_repeat']} folds.
- Holdout labels were used only in the final evaluation stage.

## Development OOF metrics

{table(oof)}

## Independent holdout metrics

{table(holdout)}

## Selective prediction

{table(selective)}

## Selected TabDistill interactions

```json
{json.dumps(interactions.get('selected', []), ensure_ascii=False, indent=2)}
```

## Final PySR formulas

```json
{json.dumps(formulas.get('final_formulas', []), ensure_ascii=False, indent=2)}
```

## Knowledge-constrained distillation

{table(pd.read_csv(distillation_path)) if distillation_path.exists() else "Not enabled for this workflow."}

## Interaction quality ablations

{table(pd.read_csv(interaction_quality_path)) if interaction_quality_path.exists() else "Not enabled for this workflow."}

## Evidence limitations

{limitation_text}
"""
    report_path = context.run_dir / "reports" / "REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    model_card_path = context.run_dir / "reports" / "MODEL_CARD.md"
    model_card_path.write_text(
        "# HemoZero model card\n\nIntended use: research-only anxiety risk classification.\n\n"
        "Inputs: six blood biomarkers and nine deterministic ratios.\n\n"
        "Not intended for diagnosis or autonomous clinical decisions. External validation is limited "
        "to the supplied fixed holdout split.\n", encoding="utf-8")
    checklist_path = context.run_dir / "reports" / "TRIPOD_AI_CHECKLIST.md"
    checklist_path.write_text(
        "# TRIPOD+AI working checklist\n\n- [x] Participant split documented\n- [x] Predictors defined\n"
        "- [x] Outcome defined\n- [x] Internal validation described\n- [x] Performance with uncertainty reported\n"
        "- [ ] Independent-site external validation\n- [ ] Prospective clinical evaluation\n", encoding="utf-8")
    paths = (report_path, model_card_path, checklist_path)
    return ToolResult("reporting", Status.COMPLETE, "Audit-gated report, model card and checklist created",
                      [ArtifactRef(str(path), sha256_file(path)) for path in paths])
