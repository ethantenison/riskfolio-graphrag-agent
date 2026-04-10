"""Materialize a promoted retrieval graph for Neo4j.

This module maps open extraction outputs, canonicalization decisions, and
schema induction artifacts into a narrower property graph intended for
retrieval. The final graph keeps evidence-linked assertions available while
avoiding uncontrolled label or relationship-type explosion.

Inputs are extraction, canonicalization, and schema induction outputs. Outputs
are constraint statements, materialized nodes and edges, representative Cypher
queries, and an optional write path into Neo4j.

This module does not perform extraction or schema induction itself.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from neo4j import GraphDatabase

from riskfolio_graphrag_agent.kg_models import (
    CanonicalizationResult,
    GraphWritePlan,
    MaterializedEdge,
    MaterializedNode,
    OpenChunkExtraction,
    ReviewStatus,
    SchemaInductionResult,
    slugify,
    stable_id,
)
from riskfolio_graphrag_agent.semantic_quality import is_semantic_assertion, is_semantic_relation_label


class GraphMaterializationPipeline:
    """Build a promoted property graph that remains retrieval-friendly."""

    def __init__(self, promotion_threshold: float = 0.6) -> None:
        """Initialize the materializer.

        Args:
            promotion_threshold: Minimum assertion confidence for promotion.
        """
        self._promotion_threshold = promotion_threshold

    def run(
        self,
        extractions: Sequence[OpenChunkExtraction],
        canonicalization: CanonicalizationResult,
        schema_induction: SchemaInductionResult,
    ) -> GraphWritePlan:
        """Build a graph write plan from upstream pipeline outputs.

        Args:
            extractions: Chunk-level open extraction outputs.
            canonicalization: Canonicalized semantic records.
            schema_induction: Stabilized schema and review artifacts.

        Returns:
            A write plan with constraints, nodes, edges, and query examples.
        """
        nodes: list[MaterializedNode] = []
        edges: list[MaterializedEdge] = []
        mention_to_entity: dict[str, str] = {}
        entity_lookup = {entity.canonical_entity_id: entity for entity in canonicalization.canonical_entities}

        for extraction in extractions:
            nodes.append(
                MaterializedNode(
                    node_id=extraction.source_document.document_id,
                    label="SourceDocument",
                    properties={
                        "source_path": extraction.source_document.source_path,
                        "source_type": extraction.source_document.source_type,
                        "title": extraction.source_document.title,
                    },
                )
            )
            nodes.append(
                MaterializedNode(
                    node_id=extraction.chunk.chunk_id,
                    label="Chunk",
                    properties={
                        "source_path": extraction.chunk.source_path,
                        "relative_path": str(extraction.chunk.metadata.get("relative_path", "")),
                        "chunk_index": extraction.chunk.chunk_index,
                        "chunk_kind": str(extraction.chunk.metadata.get("chunk_kind", extraction.chunk.section or "")),
                        "section": extraction.chunk.section,
                        "line_start": extraction.chunk.line_start,
                        "line_end": extraction.chunk.line_end,
                        "content_hash": extraction.chunk.content_hash,
                        "content": extraction.chunk.content,
                    },
                )
            )
            edges.append(
                MaterializedEdge(
                    source_id=extraction.source_document.document_id,
                    target_id=extraction.chunk.chunk_id,
                    relationship_type="HAS_CHUNK",
                )
            )

        for ontology_class in schema_induction.ontology_classes:
            nodes.append(
                MaterializedNode(
                    node_id=ontology_class.ontology_class_id,
                    label="OntologyClass",
                    properties={
                        "label": ontology_class.label,
                        "definition": ontology_class.definition,
                        "status": ontology_class.status.value,
                        "confidence": ontology_class.confidence,
                    },
                )
            )

        for ontology_property in schema_induction.ontology_properties:
            nodes.append(
                MaterializedNode(
                    node_id=ontology_property.ontology_property_id,
                    label="OntologyProperty",
                    properties={
                        "label": ontology_property.label,
                        "definition": ontology_property.definition,
                        "status": ontology_property.status.value,
                        "confidence": ontology_property.confidence,
                    },
                )
            )

        for scheme in schema_induction.concept_schemes:
            nodes.append(
                MaterializedNode(
                    node_id=scheme.concept_scheme_id,
                    label="ConceptScheme",
                    properties={"label": scheme.label, "status": scheme.status.value},
                )
            )
            for concept_id in scheme.concept_ids:
                edges.append(
                    MaterializedEdge(
                        source_id=scheme.concept_scheme_id,
                        target_id=concept_id,
                        relationship_type="HAS_CONCEPT",
                    )
                )

        class_lookup = {item.label.casefold(): item.ontology_class_id for item in schema_induction.ontology_classes}
        property_lookup = {
            self._normalize_property_label(item.label): item.ontology_property_id for item in schema_induction.ontology_properties
        }

        for entity in canonicalization.canonical_entities:
            nodes.append(
                MaterializedNode(
                    node_id=entity.canonical_entity_id,
                    label="CanonicalEntity",
                    properties={
                        "preferred_label": entity.preferred_label,
                        "normalized_label": entity.normalized_label,
                        "type_guess": entity.type_guess,
                        "confidence": entity.confidence,
                        "status": entity.status.value,
                        "provenance_chunk_ids": entity.provenance_chunk_ids,
                    },
                )
            )
            if entity.type_guess.casefold() in class_lookup:
                edges.append(
                    MaterializedEdge(
                        source_id=entity.canonical_entity_id,
                        target_id=class_lookup[entity.type_guess.casefold()],
                        relationship_type="INSTANCE_OF",
                    )
                )
            for mention_id in entity.mention_ids:
                mention_to_entity[mention_id] = entity.canonical_entity_id

        promoted_predicates = {
            item.preferred_label.casefold(): item
            for item in canonicalization.canonical_predicates
            if item.status in {ReviewStatus.PROPOSED, ReviewStatus.REVIEWED, ReviewStatus.PROMOTED}
            and is_semantic_relation_label(item.preferred_label)
        }

        for predicate in promoted_predicates.values():
            property_id = property_lookup.get(self._normalize_property_label(predicate.preferred_label))
            if property_id is None:
                continue
            edges.append(
                MaterializedEdge(
                    source_id=property_id,
                    target_id=property_id,
                    relationship_type="ALIGNS_WITH",
                    properties={"canonical_predicate_id": predicate.canonical_predicate_id},
                )
            )

        for extraction in extractions:
            for assertion in extraction.candidate_assertions:
                if assertion.confidence < self._promotion_threshold:
                    continue
                subject_id = mention_to_entity.get(assertion.subject_mention_id)
                object_id = mention_to_entity.get(assertion.object_mention_id)
                predicate = promoted_predicates.get(assertion.relation_guess.casefold())
                if subject_id is None or object_id is None or predicate is None:
                    continue
                subject_entity = entity_lookup.get(subject_id)
                object_entity = entity_lookup.get(object_id)
                if not is_semantic_assertion(
                    assertion.relation_guess,
                    subject_type=subject_entity.type_guess if subject_entity is not None else None,
                    object_type=object_entity.type_guess if object_entity is not None else None,
                ):
                    continue
                assertion_node_id = stable_id("materialized-assertion", assertion.assertion_id)
                nodes.append(
                    MaterializedNode(
                        node_id=assertion_node_id,
                        label="Assertion",
                        properties={
                            "statement": assertion.statement,
                            "relation_guess": assertion.relation_guess,
                            "confidence": assertion.confidence,
                            "status": ReviewStatus.PROMOTED.value,
                            "source_chunk_id": assertion.chunk_id,
                            "evidence_ids": assertion.evidence_ids,
                        },
                    )
                )
                edges.extend(
                    [
                        MaterializedEdge(
                            source_id=assertion_node_id,
                            target_id=subject_id,
                            relationship_type="ASSERTS_SUBJECT",
                        ),
                        MaterializedEdge(
                            source_id=assertion_node_id,
                            target_id=object_id,
                            relationship_type="ASSERTS_OBJECT",
                        ),
                        MaterializedEdge(
                            source_id=assertion_node_id,
                            target_id=property_lookup[self._normalize_property_label(predicate.preferred_label)],
                            relationship_type="ASSERTS_PREDICATE",
                        ),
                        MaterializedEdge(
                            source_id=assertion_node_id,
                            target_id=assertion.chunk_id,
                            relationship_type="SUPPORTED_BY",
                        ),
                    ]
                )

        return GraphWritePlan(
            constraints=self._build_constraints(),
            nodes=self._dedupe_nodes(nodes),
            edges=self._dedupe_edges(edges),
            retrieval_queries=self._build_retrieval_queries(),
        )

    def _normalize_property_label(self, label: str) -> str:
        return slugify(label)

    def _build_constraints(self) -> list[str]:
        return [
            "CREATE CONSTRAINT source_document_id IF NOT EXISTS FOR (n:SourceDocument) REQUIRE n.node_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (n:Chunk) REQUIRE n.node_id IS UNIQUE",
            "CREATE CONSTRAINT canonical_entity_id IF NOT EXISTS FOR (n:CanonicalEntity) REQUIRE n.node_id IS UNIQUE",
            "CREATE CONSTRAINT assertion_id IF NOT EXISTS FOR (n:Assertion) REQUIRE n.node_id IS UNIQUE",
            "CREATE CONSTRAINT ontology_class_id IF NOT EXISTS FOR (n:OntologyClass) REQUIRE n.node_id IS UNIQUE",
            "CREATE CONSTRAINT ontology_property_id IF NOT EXISTS FOR (n:OntologyProperty) REQUIRE n.node_id IS UNIQUE",
            "CREATE INDEX canonical_entity_label IF NOT EXISTS FOR (n:CanonicalEntity) ON (n.preferred_label)",
            "CREATE INDEX assertion_relation IF NOT EXISTS FOR (n:Assertion) ON (n.relation_guess)",
        ]

    def _build_retrieval_queries(self) -> dict[str, str]:
        return {
            "entity_context": (
                "MATCH (e:CanonicalEntity)-[:INSTANCE_OF]->(c:OntologyClass) "
                "WHERE toLower(e.preferred_label) CONTAINS toLower($query) "
                "RETURN e, c LIMIT $limit"
            ),
            "assertion_evidence": (
                "MATCH (a:Assertion)-[:ASSERTS_SUBJECT]->(s:CanonicalEntity), "
                "(a)-[:ASSERTS_OBJECT]->(o:CanonicalEntity), "
                "(a)-[:SUPPORTED_BY]->(c:Chunk) "
                "WHERE toLower(s.preferred_label) CONTAINS toLower($query) "
                "OR toLower(o.preferred_label) CONTAINS toLower($query) "
                "RETURN a, s, o, c ORDER BY a.confidence DESC LIMIT $limit"
            ),
            "ontology_guided_neighbors": (
                "MATCH (e:CanonicalEntity)-[:INSTANCE_OF]->(oc:OntologyClass)<-[:HAS_CONCEPT]-(:ConceptScheme) "
                "WHERE toLower(oc.label) CONTAINS toLower($query) "
                "RETURN e, oc LIMIT $limit"
            ),
        }

    def _dedupe_nodes(self, nodes: Iterable[MaterializedNode]) -> list[MaterializedNode]:
        deduped: dict[tuple[str, str], MaterializedNode] = {}
        for node in nodes:
            deduped[(node.label, node.node_id)] = node
        return list(deduped.values())

    def _dedupe_edges(self, edges: Iterable[MaterializedEdge]) -> list[MaterializedEdge]:
        deduped: dict[tuple[str, str, str], MaterializedEdge] = {}
        for edge in edges:
            deduped[(edge.source_id, edge.relationship_type, edge.target_id)] = edge
        return list(deduped.values())


def write_materialized_graph(
    *,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    write_plan: GraphWritePlan,
    drop_existing: bool = False,
) -> None:
    """Apply a materialized graph write plan to Neo4j.

    Args:
        neo4j_uri: Bolt URI for Neo4j.
        neo4j_user: Neo4j username.
        neo4j_password: Neo4j password.
        write_plan: Materialized graph write plan.
        drop_existing: Whether to clear the database before writing.
    """
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    try:
        with driver.session() as session:
            if drop_existing:
                session.run("MATCH (n) DETACH DELETE n").consume()
            for statement in write_plan.constraints:
                session.run(statement).consume()
            for label in sorted({node.label for node in write_plan.nodes}):
                rows = [
                    {"node_id": node.node_id, "properties": {**node.properties, "node_id": node.node_id}}
                    for node in write_plan.nodes
                    if node.label == label
                ]
                if not rows:
                    continue
                session.run(
                    f"UNWIND $rows AS row MERGE (n:{label} {{node_id: row.node_id}}) SET n += row.properties",
                    rows=rows,
                ).consume()
            for relationship_type in sorted({edge.relationship_type for edge in write_plan.edges}):
                rows = [
                    {
                        "source_id": edge.source_id,
                        "target_id": edge.target_id,
                        "properties": edge.properties,
                    }
                    for edge in write_plan.edges
                    if edge.relationship_type == relationship_type
                ]
                if not rows:
                    continue
                session.run(
                    (
                        "UNWIND $rows AS row "
                        "MATCH (s {node_id: row.source_id}) MATCH (t {node_id: row.target_id}) "
                        f"MERGE (s)-[r:{relationship_type}]->(t) SET r += row.properties"
                    ),
                    rows=rows,
                ).consume()
    finally:
        driver.close()
