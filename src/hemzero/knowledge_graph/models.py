"""Typed records used by the knowledge-constrained distillation layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class InteractionStatus(StrEnum):
    """Auditable policy outcomes for a candidate feature interaction."""

    REJECT_MECHANICAL = "reject_mechanical"
    REJECT_UNSTABLE = "reject_unstable"
    RETAIN_SUPPORTED = "retain_supported"
    RETAIN_DISCOVERY = "retain_discovery"
    REVIEW_CONFLICT = "review_conflict"
    REJECT_INSUFFICIENT = "reject_insufficient"


@dataclass(frozen=True)
class InteractionAssessment:
    """All facts available to policy; Neo4j does not derive the decision."""

    direct_derivation: bool
    shares_component: bool
    selection_frequency: float | None
    direction_consistency: float | None
    has_supporting_evidence: bool
    has_conflicting_evidence: bool
    supporting_assertion_ids: tuple[str, ...] = ()
    conflicting_assertion_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class InteractionDecision:
    """Persistable output of one deterministic policy evaluation."""

    decision_id: str
    run_id: str
    summary_id: str
    pair_id: str
    feature_a: str
    feature_b: str
    status: InteractionStatus
    reason: str
    assessment: InteractionAssessment
    policy_version: str
    repeat: int | None = None
    fold: int | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        return value


@dataclass(frozen=True)
class EvidenceMatch:
    """Direct external evidence retrieved for a canonical feature pair."""

    supporting_assertion_ids: tuple[str, ...] = ()
    conflicting_assertion_ids: tuple[str, ...] = ()

    @property
    def has_support(self) -> bool:
        return bool(self.supporting_assertion_ids)

    @property
    def has_conflict(self) -> bool:
        return bool(self.conflicting_assertion_ids)


@dataclass(frozen=True)
class AuditFinding:
    """One machine-readable knowledge-graph audit result."""

    check: str
    passed: bool
    detail: str
    metadata: dict[str, Any] = field(default_factory=dict)
