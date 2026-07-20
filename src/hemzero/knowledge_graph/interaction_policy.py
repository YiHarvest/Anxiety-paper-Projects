"""Rule-based policy for knowledge-constrained feature-interaction decisions."""

from __future__ import annotations

from .models import InteractionAssessment, InteractionStatus


def decide_interaction(
    assessment: InteractionAssessment,
    *,
    min_frequency: float,
    min_direction_consistency: float,
) -> InteractionStatus:
    """Apply rules in a fixed, auditable order without a synthetic KG score."""

    if assessment.direct_derivation or assessment.shares_component:
        return InteractionStatus.REJECT_MECHANICAL
    if assessment.selection_frequency is None or assessment.direction_consistency is None:
        return InteractionStatus.REJECT_INSUFFICIENT
    if not 0 <= assessment.selection_frequency <= 1 or not 0 <= assessment.direction_consistency <= 1:
        return InteractionStatus.REJECT_INSUFFICIENT
    if (
        assessment.selection_frequency < min_frequency
        or assessment.direction_consistency < min_direction_consistency
    ):
        return InteractionStatus.REJECT_UNSTABLE
    if assessment.has_conflicting_evidence:
        return InteractionStatus.REVIEW_CONFLICT
    if assessment.has_supporting_evidence:
        return InteractionStatus.RETAIN_SUPPORTED
    return InteractionStatus.RETAIN_DISCOVERY


def decision_reason(status: InteractionStatus, assessment: InteractionAssessment) -> str:
    reasons = {
        InteractionStatus.REJECT_MECHANICAL: "direct derivation or shared deterministic component",
        InteractionStatus.REJECT_UNSTABLE: "selection frequency or direction consistency below policy threshold",
        InteractionStatus.RETAIN_SUPPORTED: "stable, non-mechanical interaction with direct accepted external evidence",
        InteractionStatus.RETAIN_DISCOVERY: "stable, non-mechanical interaction without direct external evidence",
        InteractionStatus.REVIEW_CONFLICT: "stable, non-mechanical interaction with conflicting accepted evidence",
        InteractionStatus.REJECT_INSUFFICIENT: "missing or invalid stability measurements",
    }
    return reasons[status]
