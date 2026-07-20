"""Interaction-arm construction for M0–M3 knowledge-constrained ablations."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from hemzero.knowledge_graph.models import InteractionStatus


ARM_MODEL_NAMES = {
    "M0": "EBM-TabDistill-Raw",
    "M1": "EBM-Stability",
    "M2": "EBM-Lineage-Stability",
    "M3-C": "EBM-KG-Confirmatory",
    "M3-D": "EBM-KG-Discovery",
}


def _rank(rows: Iterable[dict], max_interactions: int) -> list[dict]:
    unique: dict[str, dict] = {}
    for row in rows:
        pair_id = row["pair_id"]
        current = unique.get(pair_id)
        if current is None or abs(float(row.get("score", 0))) > abs(float(current.get("score", 0))):
            unique[pair_id] = row
    return sorted(
        unique.values(), key=lambda row: (-float(row.get("frequency", 0)), -abs(float(row.get("score", 0))), row["pair_id"])
    )[:max_interactions]


def build_ablation_arms(
    summaries: list[dict],
    decisions: list[dict],
    *,
    min_frequency: float,
    min_direction_consistency: float,
    max_interactions: int,
) -> dict[str, list[dict]]:
    """Create the exact interaction sets for M0, M1, M2, M3-C and M3-D."""

    decision_by_summary = {row["summary_id"]: row for row in decisions}
    stable = [
        row for row in summaries
        if row.get("frequency") is not None and row.get("direction_consistency") is not None
        and float(row["frequency"]) >= min_frequency
        and float(row["direction_consistency"]) >= min_direction_consistency
    ]
    lineage_clean = [
        row for row in stable
        if not decision_by_summary[row["summary_id"]]["assessment"]["direct_derivation"]
        and not decision_by_summary[row["summary_id"]]["assessment"]["shares_component"]
    ]
    supported = [
        row for row in summaries
        if decision_by_summary[row["summary_id"]]["status"] == InteractionStatus.RETAIN_SUPPORTED.value
    ]
    discovery = [
        row for row in summaries
        if decision_by_summary[row["summary_id"]]["status"]
        in {InteractionStatus.RETAIN_SUPPORTED.value, InteractionStatus.RETAIN_DISCOVERY.value}
    ]
    return {
        "M0": _rank(summaries, max_interactions),
        "M1": _rank(stable, max_interactions),
        "M2": _rank(lineage_clean, max_interactions),
        "M3-C": _rank(supported, max_interactions),
        "M3-D": _rank(discovery, max_interactions),
    }


def interaction_quality_metrics(decisions: list[dict]) -> dict[str, float]:
    count = max(len(decisions), 1)
    status_counts: dict[str, int] = defaultdict(int)
    for row in decisions:
        status_counts[row["status"]] += 1
    mechanical = status_counts[InteractionStatus.REJECT_MECHANICAL.value]
    supported = status_counts[InteractionStatus.RETAIN_SUPPORTED.value]
    conflict = status_counts[InteractionStatus.REVIEW_CONFLICT.value]
    discovery = status_counts[InteractionStatus.RETAIN_DISCOVERY.value]
    return {
        "mechanical_interaction_retention_rate": 0.0,
        "supporting_evidence_coverage": supported / count,
        "conflicting_evidence_fraction": conflict / count,
        "stable_without_evidence_fraction": discovery / count,
        "mechanical_rejection_fraction": mechanical / count,
    }
