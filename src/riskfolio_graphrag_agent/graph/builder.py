"""Knowledge graph builder – writes entities and relationships to Neo4j.

Responsibilities
----------------
1. Accept a list of :class:`~riskfolio_graphrag_agent.ingestion.loader.Document`
   objects.
2. Extract entities (functions, classes, parameters, concepts) from each chunk.
3. Upsert nodes and edges into Neo4j using the Bolt driver.

This module currently provides **stub** implementations.  Replace the
``# TODO`` sections with real NLP/LLM-based extraction logic.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from neo4j import Driver, GraphDatabase
from neo4j.exceptions import Neo4jError

from riskfolio_graphrag_agent.ingestion.loader import Document

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """A single node to be upserted into the knowledge graph.

    Attributes:
        label: Neo4j node label (e.g. "Function", "Class", "Concept").
        name: Unique identifier for this node within its label.
        properties: Additional key/value properties stored on the node.
    """

    label: str
    name: str
    properties: dict[str, str] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """A directed relationship between two nodes.

    Attributes:
        source_name: Name of the source node.
        target_name: Name of the target node.
        relation_type: Neo4j relationship type string (e.g. "CALLS", "DOCUMENTS").
        properties: Additional key/value properties on the relationship.
    """

    source_name: str
    target_name: str
    relation_type: str
    source_label: str = "Entity"
    target_label: str = "Entity"
    properties: dict[str, str] = field(default_factory=dict)


class GraphBuilder:
    """Coordinates entity extraction and Neo4j upserts.

    Args:
        neo4j_uri: Bolt URI for the Neo4j instance.
        neo4j_user: Neo4j username.
        neo4j_password: Neo4j password.
    """

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str) -> None:
        self._uri = neo4j_uri
        self._user = neo4j_user
        self._password = neo4j_password
        self._driver: Driver | None = None

    def _ensure_driver(self) -> Driver:
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password),
            )
            self._driver.verify_connectivity()
        return self._driver

    def close(self) -> None:
        """Close the underlying Neo4j driver connection."""
        if self._driver is not None:
            self._driver.close()

    def build(self, documents: list[Document], drop_existing: bool = False) -> None:
        """Extract entities from *documents* and write them to Neo4j.

        Args:
            documents: Chunked source documents produced by the ingestion loader.
            drop_existing: When ``True``, delete all nodes and edges before
                re-building the graph.
        """
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []

        for doc in documents:
            doc_nodes, doc_edges = _extract_entities(doc)
            nodes.extend(doc_nodes)
            edges.extend(doc_edges)

        logger.info("Extracted %d nodes and %d edges.", len(nodes), len(edges))

        try:
            driver = self._ensure_driver()
        except (Neo4jError, OSError) as exc:
            logger.warning("Neo4j unavailable; skipping graph writes: %s", exc)
            return

        with driver.session() as session:
            if drop_existing:
                logger.warning("drop_existing=True – wiping graph.")
                session.run("MATCH (n) DETACH DELETE n")
            _upsert_nodes(session, nodes)
            _upsert_edges(session, edges)

    def get_stats(self) -> dict[str, int | dict[str, int]]:
        """Return simple graph statistics from Neo4j.

        Returns:
            A dictionary containing total node/relationship counts, node
            counts grouped by label, and relationship counts grouped by type.

        Raises:
            Neo4jError: If the query fails.
            OSError: If a transport/connectivity issue occurs.
        """
        driver = self._ensure_driver()
        with driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) AS count").single()
            relationship_count = session.run(
                "MATCH ()-[r]->() RETURN count(r) AS count"
            ).single()
            label_rows = list(
                session.run(
                "MATCH (n) UNWIND labels(n) AS label "
                "RETURN label, count(*) AS count ORDER BY count DESC"
                )
            )
            relationship_rows = list(
                session.run(
                    "MATCH ()-[r]->() "
                    "RETURN type(r) AS relationship_type, count(*) AS count "
                    "ORDER BY count DESC"
                )
            )

        node_counts_by_label: dict[str, int] = {}
        for row in label_rows:
            node_counts_by_label[str(row["label"])] = int(row["count"])

        relationship_counts_by_type: dict[str, int] = {}
        for row in relationship_rows:
            relationship_counts_by_type[str(row["relationship_type"])] = int(row["count"])

        return {
            "nodes": int(node_count["count"]) if node_count is not None else 0,
            "relationships": int(relationship_count["count"]) if relationship_count is not None else 0,
            "node_counts_by_label": node_counts_by_label,
            "relationship_counts_by_type": relationship_counts_by_type,
        }


# ── Private helpers ────────────────────────────────────────────────────────────


def _extract_entities(doc: Document) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Extract named entities and relationships from a single document chunk.

    Args:
        doc: A text chunk from the ingestion pipeline.

    Returns:
        A tuple of (nodes, edges) extracted from the chunk.
    """
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    source_stem = re.sub(r"[^A-Za-z0-9_]+", "_", doc.source_path.split("/")[-1])
    document_name = f"{source_stem}:{doc.chunk_index}"
    nodes.append(
        GraphNode(
            label="Document",
            name=document_name,
            properties={
                "source_path": doc.source_path,
                "chunk_index": str(doc.chunk_index),
                "extension": doc.metadata.get("extension", ""),
            },
        )
    )

    seen_functions: set[str] = set()
    seen_classes: set[str] = set()
    seen_concepts: set[str] = set()

    for match in re.finditer(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", doc.content, flags=re.M):
        func_name = match.group(1)
        if func_name in seen_functions:
            continue
        seen_functions.add(func_name)
        nodes.append(GraphNode(label="Function", name=func_name))
        edges.append(
            GraphEdge(
                source_name=document_name,
                target_name=func_name,
                source_label="Document",
                target_label="Function",
                relation_type="MENTIONS",
            )
        )

    for match in re.finditer(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\b", doc.content, flags=re.M):
        class_name = match.group(1)
        if class_name in seen_classes:
            continue
        seen_classes.add(class_name)
        nodes.append(GraphNode(label="Class", name=class_name))
        edges.append(
            GraphEdge(
                source_name=document_name,
                target_name=class_name,
                source_label="Document",
                target_label="Class",
                relation_type="MENTIONS",
            )
        )

    for line in doc.content.splitlines():
        heading = line.strip()
        if not heading:
            continue
        if heading.startswith("#"):
            concept_name = heading.lstrip("#").strip()
        elif set(heading) <= {"=", "-", "~", "^"}:
            continue
        else:
            continue

        if not concept_name or concept_name in seen_concepts:
            continue
        seen_concepts.add(concept_name)
        nodes.append(GraphNode(label="Concept", name=concept_name))
        edges.append(
            GraphEdge(
                source_name=document_name,
                target_name=concept_name,
                source_label="Document",
                target_label="Concept",
                relation_type="MENTIONS",
            )
        )

    return nodes, edges


def _upsert_nodes(session, nodes: list[GraphNode]) -> None:
    """Upsert a batch of nodes into Neo4j.

    Args:
        nodes: Nodes to create-or-update.
    """
    if not nodes:
        return

    nodes_by_label: dict[str, list[GraphNode]] = {}
    for node in nodes:
        nodes_by_label.setdefault(node.label, []).append(node)

    for label, labeled_nodes in nodes_by_label.items():
        safe_label = re.sub(r"[^A-Za-z0-9_]", "", label) or "Entity"
        cypher = (
            f"UNWIND $rows AS row "
            f"MERGE (n:{safe_label} {{name: row.name}}) "
            "SET n += row.properties"
        )
        session.run(
            cypher,
            rows=[{"name": n.name, "properties": n.properties} for n in labeled_nodes],
        )


def _upsert_edges(session, edges: list[GraphEdge]) -> None:
    """Upsert a batch of edges into Neo4j.

    Args:
        edges: Edges to create-or-update.
    """
    if not edges:
        return

    for edge in edges:
        safe_source_label = re.sub(r"[^A-Za-z0-9_]", "", edge.source_label) or "Entity"
        safe_target_label = re.sub(r"[^A-Za-z0-9_]", "", edge.target_label) or "Entity"
        safe_relation = re.sub(r"[^A-Za-z0-9_]", "", edge.relation_type) or "RELATED_TO"
        cypher = (
            f"MATCH (s:{safe_source_label} {{name: $source_name}}) "
            f"MATCH (t:{safe_target_label} {{name: $target_name}}) "
            f"MERGE (s)-[r:{safe_relation}]->(t) "
            "SET r += $properties"
        )
        session.run(
            cypher,
            source_name=edge.source_name,
            target_name=edge.target_name,
            properties=edge.properties,
        )
