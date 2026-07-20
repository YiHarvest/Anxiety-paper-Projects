from __future__ import annotations

from hemzero.common.hashing import sha256_file
from hemzero.common.io import atomic_json
from hemzero.common.schemas import ArtifactRef, Status, ToolResult
from hemzero.distillation.knowledge_constrained_distillation import build_ablation_arms, interaction_quality_metrics
from hemzero.graph.lineage import build_lineage_graph
from hemzero.knowledge_graph.interaction_policy import decision_reason
from hemzero.knowledge_graph.interaction_query import assess_interaction
from hemzero.knowledge_graph.models import InteractionDecision, InteractionStatus
from hemzero.knowledge_graph.snapshot import read_jsonl, write_jsonl
from pi_agent.policy_engine import PolicyEngine

from ._kg_shared import open_repository
from ._shared import ToolContext, require_file


def run(arguments: dict, context: ToolContext) -> ToolResult:
    policy_config = context.config("interaction_policy.yaml")
    summaries = read_jsonl(require_file(context.run_dir / "knowledge_graph" / "interaction_summaries.jsonl", "interaction_summaries"))
    dataset = context.config("dataset.yaml")
    graph = build_lineage_graph(dataset["raw_features"], context.config("ratio_formulas.yaml"))
    decisions: list[dict] = []
    try:
        with open_repository(context) as repository:
            for summary in summaries:
                assessment = assess_interaction(summary, graph, repository)
                status = PolicyEngine.decide_interaction(assessment, policy_config)
                repeat, fold = summary.get("repeat"), summary.get("fold")
                decision = InteractionDecision(
                    decision_id=f"{summary['summary_id']}__decision", run_id=context.run_dir.name,
                    summary_id=summary["summary_id"], pair_id=summary["pair_id"],
                    feature_a=summary["feature_a"], feature_b=summary["feature_b"], status=status,
                    reason=decision_reason(status, assessment), assessment=assessment,
                    policy_version=str(policy_config["version"]), repeat=repeat, fold=fold,
                ).to_dict()
                decisions.append(decision)
            repository.upsert_decisions(decisions)
    except Exception as exc:
        return ToolResult("kg_assess_interactions", Status.BLOCKED, f"Knowledge-constrained assessment failed: {exc}")
    output = context.run_dir / "knowledge_graph" / "interaction_decisions.jsonl"
    write_jsonl(output, decisions)
    global_summaries = [row for row in summaries if row["scope"] == "development_global"]
    global_ids = {row["summary_id"] for row in global_summaries}
    global_decisions = [row for row in decisions if row["summary_id"] in global_ids]
    arms = build_ablation_arms(
        global_summaries, global_decisions, min_frequency=float(policy_config["min_frequency"]),
        min_direction_consistency=float(policy_config["min_direction_consistency"]),
        max_interactions=int(policy_config["max_interactions"]),
    )
    legacy_selection = {
        "selected": arms["M3-D"], "confirmatory": arms["M3-C"], "discovery": arms["M3-D"],
        "conflicted": [row for row in global_decisions if row["status"] == InteractionStatus.REVIEW_CONFLICT.value],
        "decisions": global_decisions, "policy_version": policy_config["version"],
    }
    selection_path = context.run_dir / "evidence" / "tabdistill_interaction_selection.json"
    atomic_json(selection_path, legacy_selection)
    conflict_path = context.run_dir / "reports" / "CONFLICTED_INTERACTIONS.md"
    conflicts = legacy_selection["conflicted"]
    conflict_path.write_text(
        "# Conflicted interactions\n\n" + ("No conflicted interactions.\n" if not conflicts else
        "\n".join(f"- `{row['feature_a']} × {row['feature_b']}`: {row['reason']}" for row in conflicts) + "\n"),
        encoding="utf-8",
    )
    artifacts = [ArtifactRef(str(path), sha256_file(path)) for path in (output, selection_path, conflict_path)]
    return ToolResult("kg_assess_interactions", Status.COMPLETE,
                      "PolicyEngine classified interactions after lineage, stability and direct evidence checks",
                      artifacts, {"decisions": len(decisions), "global_arms": {k: len(v) for k, v in arms.items()},
                                  "quality": interaction_quality_metrics(global_decisions),
                                  "neo4j_made_decisions": False, "holdout_labels_accessed": False})
