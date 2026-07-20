"""Official Neo4j-driver repository with idempotent, parameterized writes."""

from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass
from typing import Any

from .models import EvidenceMatch
from .schema import CONSTRAINT_QUERIES, DETERMINISTIC_RELATIONSHIPS


@dataclass(frozen=True)
class Neo4jSettings:
    uri: str
    username: str
    password: str
    database: str = "neo4j"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Neo4jSettings":
        driver = config["driver"]
        values = {
            "uri": os.getenv(driver["uri_env"]),
            "username": os.getenv(driver["username_env"]),
            "password": os.getenv(driver["password_env"]),
            "database": os.getenv(driver["database_env"], driver.get("default_database", "neo4j")),
        }
        missing = [name for name in ("uri", "username", "password") if not values[name]]
        if missing:
            raise ValueError(f"Missing Neo4j environment settings: {missing}")
        return cls(**values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class Neo4jCapability:
    available: bool
    configured: bool
    reason: str


def capability(config: dict[str, Any]) -> Neo4jCapability:
    if importlib.util.find_spec("neo4j") is None:
        return Neo4jCapability(False, False, "neo4j driver is not installed; use uv sync --extra knowledge-graph")
    try:
        Neo4jSettings.from_config(config)
    except ValueError as exc:
        return Neo4jCapability(True, False, str(exc))
    return Neo4jCapability(True, True, "neo4j driver and connection settings detected")


def _properties(value: dict[str, Any]) -> dict[str, Any]:
    """Convert nested values to Neo4j-supported scalar/list properties."""

    clean: dict[str, Any] = {}
    for key, item in value.items():
        if item is None:
            continue
        if isinstance(item, (str, int, float, bool)):
            clean[key] = item
        elif isinstance(item, (list, tuple)) and all(isinstance(v, (str, int, float, bool)) for v in item):
            clean[key] = list(item)
        else:
            clean[key] = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
    return clean


LINEAGE_QUERIES = {
    "NUMERATOR_OF": """
        UNWIND $rows AS row MATCH (s:FeatureEntity {id: row.source}) MATCH (t:FeatureEntity {id: row.target})
        MERGE (s)-[r:NUMERATOR_OF {edge_id: row.edge_id}]->(t) SET r += row.properties
    """,
    "DENOMINATOR_OF": """
        UNWIND $rows AS row MATCH (s:FeatureEntity {id: row.source}) MATCH (t:FeatureEntity {id: row.target})
        MERGE (s)-[r:DENOMINATOR_OF {edge_id: row.edge_id}]->(t) SET r += row.properties
    """,
    "DERIVED_FROM": """
        UNWIND $rows AS row MATCH (s:FeatureEntity {id: row.source}) MATCH (t:FeatureEntity {id: row.target})
        MERGE (s)-[r:DERIVED_FROM {edge_id: row.edge_id}]->(t) SET r += row.properties
    """,
    "SHARES_COMPONENT": """
        UNWIND $rows AS row MATCH (s:FeatureEntity {id: row.source}) MATCH (t:FeatureEntity {id: row.target})
        MERGE (s)-[r:SHARES_COMPONENT {edge_id: row.edge_id}]->(t) SET r += row.properties
    """,
}

EXPORT_NODE_QUERIES = {
    label: f"MATCH (n:{label}) RETURN properties(n) AS properties ORDER BY coalesce(n.id, n.study_id, n.assertion_id, n.run_id, n.fold_id, n.candidate_id, n.summary_id, n.decision_id)"
    for label in (
        "Biomarker", "DerivedRatio", "BiologicalSystem", "Disease", "Study", "EvidenceAssertion",
        "ModelRun", "CVFold", "Model", "InteractionCandidate", "InteractionSummary", "InteractionDecision",
    )
}

RUN_SCOPED_EXPORT_QUERIES = {
    "ModelRun": "MATCH (n:ModelRun {run_id: $run_id}) RETURN properties(n) AS properties ORDER BY n.run_id",
    "CVFold": "MATCH (n:CVFold {run_id: $run_id}) RETURN properties(n) AS properties ORDER BY n.fold_id",
    "InteractionCandidate": "MATCH (n:InteractionCandidate {run_id: $run_id}) RETURN properties(n) AS properties ORDER BY n.candidate_id",
    "InteractionSummary": "MATCH (n:InteractionSummary {run_id: $run_id}) RETURN properties(n) AS properties ORDER BY n.summary_id",
    "InteractionDecision": "MATCH (n:InteractionDecision {run_id: $run_id}) RETURN properties(n) AS properties ORDER BY n.decision_id",
    "Model": """
        MATCH (n:Model)
        WHERE n.run_id = $run_id OR n.model_id IN ['TabPFN-3', 'TabDistill']
        WITH DISTINCT n
        RETURN properties(n) AS properties ORDER BY n.model_id
    """,
}

ID_SCOPED_EXPORT_QUERIES = {
    label: f"MATCH (n:{label}) WHERE n.id IN $allowed_ids RETURN properties(n) AS properties ORDER BY n.id"
    for label in ("Biomarker", "DerivedRatio", "BiologicalSystem", "Disease")
}


class Neo4jRepository:
    """Single shared driver wrapper. All dynamic values are Cypher parameters."""

    def __init__(
        self,
        settings: Neo4jSettings,
        *,
        verify: bool = True,
        evidence_registry_version: str | None = None,
        evidence_registry_empty: bool = False,
    ) -> None:
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(settings.uri, auth=(settings.username, settings.password))
        self._database = settings.database
        self._evidence_registry_version = evidence_registry_version
        self._evidence_registry_empty = evidence_registry_empty
        if verify:
            self._driver.verify_connectivity()

    def __enter__(self) -> "Neo4jRepository":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._driver.close()

    def _execute(self, query: str, **parameters: Any) -> list[dict[str, Any]]:
        records, _, _ = self._driver.execute_query(
            query, parameters_=parameters, database_=self._database
        )
        return [record.data() for record in records]

    def initialize_schema(self) -> None:
        for query in CONSTRAINT_QUERIES:
            self._execute(query)

    def server_version(self) -> str:
        rows = self._execute("CALL dbms.components() YIELD versions RETURN versions[0] AS version")
        return str(rows[0]["version"]) if rows else "unknown"

    def upsert_biomarkers(self, rows: list[dict[str, Any]]) -> None:
        self._execute("""
            UNWIND $rows AS row MERGE (n:Biomarker:FeatureEntity {id: row.id}) SET n += row.properties
        """, rows=[{"id": r["id"], "properties": _properties(r.get("properties", r))} for r in rows])

    def upsert_ratios(self, rows: list[dict[str, Any]]) -> None:
        self._execute("""
            UNWIND $rows AS row MERGE (n:DerivedRatio:FeatureEntity {id: row.id}) SET n += row.properties
        """, rows=[{"id": r["id"], "properties": _properties(r.get("properties", r))} for r in rows])

    def upsert_systems(self, rows: list[dict[str, Any]]) -> None:
        self._execute("""
            UNWIND $rows AS row MERGE (n:BiologicalSystem {id: row.id}) SET n += row.properties
        """, rows=[{"id": r["id"], "properties": _properties(r)} for r in rows])

    def upsert_diseases(self, rows: list[dict[str, Any]]) -> None:
        self._execute("""
            UNWIND $rows AS row MERGE (n:Disease {id: row.id}) SET n += row.properties
        """, rows=[{"id": r["id"], "properties": _properties(r)} for r in rows])

    def link_biomarker_systems(self, rows: list[dict[str, Any]]) -> None:
        self._execute("""
            UNWIND $rows AS row MATCH (b:Biomarker {id: row.biomarker_id})
            MATCH (s:BiologicalSystem {id: row.system_id}) MERGE (b)-[:PARTICIPATES_IN]->(s)
        """, rows=rows)

    def link_feature_systems(self, rows: list[dict[str, Any]]) -> None:
        self._execute("""
            UNWIND $rows AS row MATCH (feature:FeatureEntity {id: row.feature_id})
            MATCH (system:BiologicalSystem {id: row.system_id})
            MERGE (feature)-[:PARTICIPATES_IN]->(system)
        """, rows=rows)

    def sync_lineage_relationships(self, rows: list[dict[str, Any]]) -> None:
        grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in DETERMINISTIC_RELATIONSHIPS}
        for row in rows:
            relation = str(row["relationship_type"]).upper()
            if relation not in LINEAGE_QUERIES:
                raise ValueError(f"Unsupported deterministic relationship: {relation}")
            grouped[relation].append({**row, "properties": _properties(row.get("properties", {}))})
        for relation, values in grouped.items():
            if values:
                self._execute(LINEAGE_QUERIES[relation], rows=values)

    def upsert_run_and_folds(self, run: dict[str, Any], folds: list[dict[str, Any]]) -> None:
        self._execute("MERGE (r:ModelRun {run_id: $run_id}) SET r += $properties",
                      run_id=run["run_id"], properties=_properties(run))
        if run.get("target_disease_id"):
            self._execute("""
                MATCH (r:ModelRun {run_id: $run_id})
                MATCH (d:Disease {id: $disease_id})
                MERGE (r)-[p:PREDICTS]->(d)
                SET p.outcome_column = $outcome_column
            """, run_id=run["run_id"], disease_id=run["target_disease_id"],
                 outcome_column=run.get("target_column"))
        self._execute("""
            UNWIND $rows AS row MERGE (f:CVFold {fold_id: row.fold_id}) SET f += row.properties
            WITH f, row MATCH (r:ModelRun {run_id: row.run_id}) MERGE (f)-[:PART_OF_RUN]->(r)
        """, rows=[{"fold_id": r["fold_id"], "run_id": r["run_id"], "properties": _properties(r)} for r in folds])

    def upsert_candidates(self, rows: list[dict[str, Any]]) -> None:
        self._execute("MERGE (m:Model {model_id: 'TabDistill'}) SET m.name = 'TabDistill'",)
        self._execute("MERGE (m:Model {model_id: 'TabPFN-3'}) SET m.name = 'TabPFN-3'")
        self._execute("""
            MATCH (student:Model {model_id: 'TabDistill'}) MATCH (teacher:Model {model_id: 'TabPFN-3'})
            MERGE (student)-[:DISTILLED_FROM]->(teacher)
        """)
        prepared = [{**row, "properties": _properties(row)} for row in rows]
        self._execute("""
            UNWIND $rows AS row MERGE (c:InteractionCandidate {candidate_id: row.candidate_id}) SET c += row.properties
            WITH c, row MATCH (a:FeatureEntity {id: row.feature_a_id}) MATCH (b:FeatureEntity {id: row.feature_b_id})
            MATCH (f:CVFold {fold_id: row.fold_id}) MATCH (m:Model {model_id: 'TabDistill'})
            MERGE (c)-[:LEFT_FEATURE]->(a) MERGE (c)-[:RIGHT_FEATURE]->(b)
            MERGE (c)-[:GENERATED_IN]->(f) MERGE (c)-[:SELECTED_BY]->(m)
        """, rows=prepared)

    def upsert_ebm_models(self, run_id: str, arms: dict[str, list[dict[str, Any]]], model_names: dict[str, str]) -> None:
        rows = []
        for arm, summaries in arms.items():
            model_id = f"{run_id}::{model_names[arm]}"
            rows.append({"model_id": model_id, "run_id": run_id, "arm": arm, "name": model_names[arm],
                         "summary_ids": [row["summary_id"] for row in summaries]})
        self._execute("""
            UNWIND $rows AS row MERGE (model:Model {model_id: row.model_id})
            SET model.name = row.name, model.arm = row.arm, model.run_id = row.run_id
            WITH model, row MATCH (run:ModelRun {run_id: row.run_id})
            MERGE (model)-[:TRAINED_IN]->(run)
            WITH model, row UNWIND row.summary_ids AS summary_id
            MATCH (summary:InteractionSummary {summary_id: summary_id})
            MERGE (summary)-[:CONTRIBUTES_TO_PREDICTION {arm: row.arm}]->(model)
        """, rows=rows)
        self._execute("""
            MATCH (teacher:Model {model_id: 'TabPFN-3'})
            MATCH (student:Model {run_id: $run_id})
            MERGE (student)-[:DISTILLED_FROM]->(teacher)
        """, run_id=run_id)

    def upsert_studies_and_assertions(self, assertions: list[dict[str, Any]]) -> None:
        studies = {row["study"]["study_id"]: row["study"] for row in assertions}
        self._execute("""
            UNWIND $rows AS row MERGE (s:Study {study_id: row.study_id}) SET s += row.properties
        """, rows=[{"study_id": key, "properties": _properties(value)} for key, value in studies.items()])
        prepared = []
        for row in assertions:
            assertion_properties = {k: v for k, v in row.items() if k not in {"subject", "object", "study"}}
            assertion_properties.update({
                "subject_id": row["subject"]["id"], "object_id": row["object"]["id"],
                "study_id": row["study"]["study_id"],
            })
            prepared.append({
                "assertion_id": row["assertion_id"], "subject_id": row["subject"]["id"],
                "object_id": row["object"]["id"], "study_id": row["study"]["study_id"],
                "properties": _properties(assertion_properties),
            })
        self._execute("""
            UNWIND $rows AS row MERGE (a:EvidenceAssertion {assertion_id: row.assertion_id}) SET a += row.properties
            WITH a, row MATCH (subject {id: row.subject_id}) MATCH (object {id: row.object_id})
            MATCH (study:Study {study_id: row.study_id})
            MERGE (a)-[:HAS_SUBJECT]->(subject) MERGE (a)-[:HAS_OBJECT]->(object)
            MERGE (a)-[:SUPPORTED_BY]->(study)
        """, rows=prepared)

    def direct_evidence(self, feature_a_id: str, feature_b_id: str) -> EvidenceMatch:
        if self._evidence_registry_empty:
            return EvidenceMatch()
        rows = self._execute("""
            MATCH (a:EvidenceAssertion)-[:HAS_SUBJECT]->(left {id: $feature_a_id})
            MATCH (a)-[:HAS_OBJECT]->(right {id: $feature_b_id})
            WHERE coalesce(a.accepted, false) = true AND a.registry_version = $registry_version
            RETURN a.assertion_id AS assertion_id, coalesce(a.evidence_role, 'supporting') AS evidence_role
            UNION
            MATCH (a:EvidenceAssertion)-[:HAS_SUBJECT]->(left {id: $feature_b_id})
            MATCH (a)-[:HAS_OBJECT]->(right {id: $feature_a_id})
            WHERE coalesce(a.accepted, false) = true AND a.registry_version = $registry_version
            RETURN a.assertion_id AS assertion_id, coalesce(a.evidence_role, 'supporting') AS evidence_role
        """, feature_a_id=feature_a_id, feature_b_id=feature_b_id,
             registry_version=self._evidence_registry_version)
        support = sorted({str(r["assertion_id"]) for r in rows if r["evidence_role"] == "supporting"})
        conflict = sorted({str(r["assertion_id"]) for r in rows if r["evidence_role"] == "conflicting"})
        return EvidenceMatch(tuple(support), tuple(conflict))

    def upsert_summaries(self, rows: list[dict[str, Any]]) -> None:
        self._execute("""
            UNWIND $rows AS row MERGE (s:InteractionSummary {summary_id: row.summary_id}) SET s += row.properties
            WITH s, row
            MATCH (r:ModelRun {run_id: row.run_id})
            MATCH (left:FeatureEntity {id: row.feature_a_id})
            MATCH (right:FeatureEntity {id: row.feature_b_id})
            MERGE (s)-[:SUMMARIZES_RUN]->(r)
            MERGE (s)-[:LEFT_FEATURE]->(left)
            MERGE (s)-[:RIGHT_FEATURE]->(right)
        """, rows=[{**r, "properties": _properties(r)} for r in rows])

    def upsert_decisions(self, rows: list[dict[str, Any]]) -> None:
        self._execute("""
            UNWIND $rows AS row MERGE (d:InteractionDecision {decision_id: row.decision_id}) SET d += row.properties
            WITH d, row MATCH (s:InteractionSummary {summary_id: row.summary_id}) MERGE (d)-[:DECIDES_ON]->(s)
        """, rows=[{**r, "properties": _properties(r)} for r in rows])
        relationship_queries = {
            "reject_mechanical": """
                UNWIND $rows AS row MATCH (s:InteractionSummary {summary_id: row.summary_id})
                MATCH (d:InteractionDecision {decision_id: row.decision_id}) MERGE (s)-[:REJECTED_BY_LINEAGE]->(d)
            """,
            "reject_unstable": """
                UNWIND $rows AS row MATCH (s:InteractionSummary {summary_id: row.summary_id})
                MATCH (d:InteractionDecision {decision_id: row.decision_id}) MERGE (s)-[:REJECTED_BY_STABILITY]->(d)
            """,
            "reject_insufficient": """
                UNWIND $rows AS row MATCH (s:InteractionSummary {summary_id: row.summary_id})
                MATCH (d:InteractionDecision {decision_id: row.decision_id}) MERGE (s)-[:REJECTED_BY_STABILITY]->(d)
            """,
            "retain_supported": """
                UNWIND $rows AS row MATCH (s:InteractionSummary {summary_id: row.summary_id})
                MATCH (d:InteractionDecision {decision_id: row.decision_id}) MERGE (s)-[:SUPPORTED_BY_KNOWLEDGE]->(d)
            """,
            "review_conflict": """
                UNWIND $rows AS row MATCH (s:InteractionSummary {summary_id: row.summary_id})
                MATCH (d:InteractionDecision {decision_id: row.decision_id}) MERGE (s)-[:CONFLICTED_WITH_KNOWLEDGE]->(d)
            """,
        }
        for status, query in relationship_queries.items():
            selected = [row for row in rows if row["status"] == status]
            if selected:
                self._execute(query, rows=selected)

    def upsert_cohort_associations(self, rows: list[dict[str, Any]]) -> None:
        for relationship_type, query in {
            "CORRELATED_WITH": """
                UNWIND $rows AS row MATCH (a:FeatureEntity {id: row.feature_a_id}) MATCH (b:FeatureEntity {id: row.feature_b_id})
                MERGE (a)-[r:CORRELATED_WITH {association_id: row.association_id}]->(b) SET r += row.properties
            """,
            "CONDITIONALLY_ASSOCIATED_WITH": """
                UNWIND $rows AS row MATCH (a:FeatureEntity {id: row.feature_a_id}) MATCH (b:FeatureEntity {id: row.feature_b_id})
                MERGE (a)-[r:CONDITIONALLY_ASSOCIATED_WITH {association_id: row.association_id}]->(b) SET r += row.properties
            """,
        }.items():
            selected = [{**r, "properties": _properties(r)} for r in rows if r["relationship_type"] == relationship_type]
            if selected:
                self._execute(query, rows=selected)

    def export_nodes(
        self,
        label: str,
        *,
        run_id: str,
        allowed_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if label not in EXPORT_NODE_QUERIES:
            raise ValueError(f"Unsupported export label: {label}")
        if label in RUN_SCOPED_EXPORT_QUERIES:
            rows = self._execute(RUN_SCOPED_EXPORT_QUERIES[label], run_id=run_id)
        elif label == "EvidenceAssertion":
            if self._evidence_registry_empty:
                return []
            rows = self._execute("""
                MATCH (n:EvidenceAssertion {registry_version: $registry_version})
                RETURN properties(n) AS properties ORDER BY n.assertion_id
            """, registry_version=self._evidence_registry_version)
        elif label == "Study":
            if self._evidence_registry_empty:
                return []
            rows = self._execute("""
                MATCH (:EvidenceAssertion {registry_version: $registry_version})-[:SUPPORTED_BY]->(n:Study)
                WITH DISTINCT n
                RETURN properties(n) AS properties ORDER BY n.study_id
            """, registry_version=self._evidence_registry_version)
        elif allowed_ids is not None and label in ID_SCOPED_EXPORT_QUERIES:
            rows = self._execute(ID_SCOPED_EXPORT_QUERIES[label], allowed_ids=allowed_ids)
        else:
            rows = self._execute(EXPORT_NODE_QUERIES[label])
        return [row["properties"] for row in rows]

    def export_lineage_relationships(self, feature_ids: list[str]) -> list[dict[str, Any]]:
        return self._execute("""
            MATCH (s:FeatureEntity)-[r]->(t:FeatureEntity)
            WHERE type(r) IN ['NUMERATOR_OF','DENOMINATOR_OF','DERIVED_FROM','SHARES_COMPONENT']
              AND s.id IN $feature_ids AND t.id IN $feature_ids
            RETURN s.id AS source, t.id AS target, type(r) AS relationship_type, properties(r) AS properties
            ORDER BY source, target, relationship_type, properties.edge_id
        """, feature_ids=feature_ids)

    def export_run_relationships(self, run_id: str) -> list[dict[str, Any]]:
        return self._execute("""
            MATCH (source)-[r]->(target)
            WHERE (
                source:ModelRun AND source.run_id = $run_id
                AND type(r) = 'PREDICTS' AND target:Disease
            ) OR (
                source:InteractionSummary AND source.run_id = $run_id
                AND type(r) IN ['LEFT_FEATURE', 'RIGHT_FEATURE'] AND target:FeatureEntity
            )
            RETURN coalesce(source.summary_id, source.run_id) AS source,
                   coalesce(target.id, target.summary_id, target.run_id) AS target,
                   type(r) AS relationship_type, properties(r) AS properties
            ORDER BY source, relationship_type, target
        """, run_id=run_id)

    def counts(self) -> dict[str, int]:
        result = {}
        for label, query in EXPORT_NODE_QUERIES.items():
            result[label] = len(self._execute(query))
        return result
