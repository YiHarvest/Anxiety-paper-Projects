"""Knowledge-constrained distillation and Neo4j persistence layer.

NetworkX remains the computational source of truth for deterministic lineage.
Neo4j stores evidence and run provenance; policy code makes all model decisions.
"""

from .interaction_policy import decide_interaction
from .models import InteractionAssessment, InteractionDecision, InteractionStatus

__all__ = ["InteractionAssessment", "InteractionDecision", "InteractionStatus", "decide_interaction"]
