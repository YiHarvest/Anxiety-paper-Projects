"""Neo4j schema and closed vocabularies for HemoZero KG schema v1."""

from __future__ import annotations


SCHEMA_VERSION = "1.1.0"

CONSTRAINT_QUERIES = (
    "CREATE CONSTRAINT feature_entity_id IF NOT EXISTS FOR (n:FeatureEntity) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT biomarker_id IF NOT EXISTS FOR (n:Biomarker) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT ratio_id IF NOT EXISTS FOR (n:DerivedRatio) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT system_id IF NOT EXISTS FOR (n:BiologicalSystem) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT disease_id IF NOT EXISTS FOR (n:Disease) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT study_id IF NOT EXISTS FOR (n:Study) REQUIRE n.study_id IS UNIQUE",
    "CREATE CONSTRAINT assertion_id IF NOT EXISTS FOR (n:EvidenceAssertion) REQUIRE n.assertion_id IS UNIQUE",
    "CREATE CONSTRAINT run_id IF NOT EXISTS FOR (n:ModelRun) REQUIRE n.run_id IS UNIQUE",
    "CREATE CONSTRAINT fold_id IF NOT EXISTS FOR (n:CVFold) REQUIRE n.fold_id IS UNIQUE",
    "CREATE CONSTRAINT model_id IF NOT EXISTS FOR (n:Model) REQUIRE n.model_id IS UNIQUE",
    "CREATE CONSTRAINT candidate_id IF NOT EXISTS FOR (n:InteractionCandidate) REQUIRE n.candidate_id IS UNIQUE",
    "CREATE CONSTRAINT interaction_summary_id IF NOT EXISTS FOR (n:InteractionSummary) REQUIRE n.summary_id IS UNIQUE",
    "CREATE CONSTRAINT decision_id IF NOT EXISTS FOR (n:InteractionDecision) REQUIRE n.decision_id IS UNIQUE",
)

DETERMINISTIC_RELATIONSHIPS = frozenset({
    "NUMERATOR_OF", "DENOMINATOR_OF", "DERIVED_FROM", "SHARES_COMPONENT",
})
BIOLOGICAL_RELATIONSHIPS = frozenset({
    "PARTICIPATES_IN", "STIMULATES", "INHIBITS", "REGULATES", "ASSOCIATED_WITH",
})
COHORT_RELATIONSHIPS = frozenset({"CORRELATED_WITH", "CONDITIONALLY_ASSOCIATED_WITH"})
MODEL_RELATIONSHIPS = frozenset({
    "SELECTED_BY", "CONTRIBUTES_TO_PREDICTION", "DISTILLED_FROM", "REJECTED_BY_LINEAGE",
    "REJECTED_BY_STABILITY", "SUPPORTED_BY_KNOWLEDGE", "CONFLICTED_WITH_KNOWLEDGE", "PREDICTS",
})
EVIDENCE_PREDICATES = frozenset({"STIMULATES", "INHIBITS", "REGULATES", "ASSOCIATED_WITH"})


def schema_cypher() -> str:
    """Return the exact idempotent schema used for the run snapshot."""

    return ";\n\n".join(CONSTRAINT_QUERIES) + ";\n"
