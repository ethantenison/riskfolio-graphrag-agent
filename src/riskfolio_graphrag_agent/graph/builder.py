"""Build the repository knowledge graph from chunked source documents.

This module is the graph-construction entry point described in the architecture
map. It sits between the ingestion layer, which emits typed ``Document``
records, and graph-backed consumers such as retrieval, semantic export, and UI
subgraph views.

Inputs are chunked documents plus optional LLM-assisted extraction callbacks.
Outputs are Neo4j nodes and relationships, graph statistics, and bounded query
subgraphs for visualization.

Key implementation decisions:
- domain aliases are handled with deterministic regex matching so the graph has
    predictable baseline coverage even when LLM extraction is disabled;
- taxonomy edges are emitted centrally to keep ontology relationships
    consistent across ingestion runs;
- writes are idempotent through per-label unique constraints and merge-based
    upserts.

This module does not rank retrieval results, generate end-user answers, or own
HTTP or UI orchestration.

Example:
        builder = GraphBuilder("bolt://localhost:7687", "neo4j", "password")
        builder.build(documents)
        stats = builder.get_stats()
        builder.close()
"""

from __future__ import annotations

import keyword
import logging
import re
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Protocol

from neo4j import Driver, GraphDatabase
from neo4j.exceptions import Neo4jError

from riskfolio_graphrag_agent.ingestion.loader import Document

logger = logging.getLogger(__name__)

WRITE_BATCH_SIZE = 500

NODE_LABELS: tuple[str, ...] = (
    "Chunk",
    "DocPage",
    "ExampleNotebook",
    "TestCase",
    "PythonModule",
    "PythonClass",
    "PythonFunction",
    "Parameter",
    "PortfolioMethod",
    "RiskMeasure",
    "ConstraintType",
    "Estimator",
    "ReportType",
    "PlotType",
    "Solver",
    "Concept",
    "AssetClass",
    "FactorModel",
    "MarketRegime",
    "BenchmarkIndex",
    "BacktestScenario",
    "OptimizationProblem",
)

RELATIONSHIP_TYPES: tuple[str, ...] = (
    "HAS_CHUNK",
    "MENTIONS",
    "DESCRIBES",
    "DEMONSTRATES",
    "VALIDATES",
    "IMPLEMENTS",
    "DECLARES",
    "HAS_PARAMETER",
    "USES",
    "SUPPORTS_RISK_MEASURE",
    "SUPPORTS_CONSTRAINT",
    "USES_ESTIMATOR",
    "PRODUCES_REPORT",
    "RELATED_TO",
    "IS_SUBTYPE_OF",
    "ALTERNATIVE_TO",
    "REQUIRES",
    "PARAMETERIZED_BY",
    "BENCHMARKED_ON",
    "CALIBRATED_ON",
    "PRECEDES",
    "HAS_CONSTRAINT",
    "VALIDATED_AGAINST",
)

DOMAIN_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "PortfolioMethod": {
        "Hierarchical Risk Parity": ("hierarchical risk parity", "hrp"),
        "Hierarchical Equal Risk Contribution": ("hierarchical equal risk contribution", "herc"),
        "Nested Clustered Optimization": ("nested clustered optimization", "nco"),
        "Risk Parity": ("risk parity", "risk budgeting"),
        "Black Litterman": ("black litterman",),
        "Mean-Variance Optimization": ("mean variance", "mean-variance"),
        "Minimum Variance": ("minimum variance",),
        "Maximum Sharpe": ("maximum sharpe",),
        "Kelly Criterion": ("kelly criterion",),
    },
    "RiskMeasure": {
        "CVaR": (
            "cvar",
            "conditional value at risk",
            "conditional var",
            "expected shortfall",
            "es",
            "cvar95",
            "cvar99",
        ),
        "VaR": ("value at risk", "var", "value-at-risk", "var95", "var99"),
        "Semi Deviation": (
            "semi deviation",
            "semi standard deviation",
            "semi-std",
            "semidev",
            "downside deviation",
        ),
        "Semi Variance": ("semi variance", "semivariance"),
        "MAD": ("mean absolute deviation", "mad"),
        "Ulcer Index": ("ulcer index",),
        "EVaR": ("entropic value at risk", "evar"),
        "EDaR": ("entropic drawdown at risk", "edar"),
        "RLVaR": ("relativistic value at risk", "rlvar"),
        "RLDaR": ("relativistic drawdown at risk", "rldar"),
        "Tail Gini": ("tail gini",),
    },
    "FactorModel": {
        "Fama-French 3 Factor": ("fama french 3", "fama-french 3", "ff3"),
        "Fama-French 5 Factor": ("fama french 5", "fama-french 5", "ff5"),
        "Carhart 4 Factor": (
            "carhart",
            "carhart 4",
        ),
        "Macroeconomic Factor Model": ("macroeconomic factor", "macro factor"),
        "Statistical PCA Factor": ("statistical pca", "pca factor model"),
    },
    "AssetClass": {
        "Equity": ("equity", "equities", "stocks"),
        "Fixed Income": ("fixed income", "bonds", "debt"),
        "Commodity": ("commodity", "commodities"),
        "Alternatives": ("alternatives", "alternative investments", "alts"),
        "Cash": ("cash", "money market"),
    },
    "MarketRegime": {
        "Bull Market": ("bull market", "bull", "uptrend"),
        "Bear Market": ("bear market", "bear", "downtrend"),
        "High Volatility": ("high volatility", "high vol", "volatility regime"),
        "Low Correlation": ("low correlation", "decorrelated", "diversification regime"),
    },
    "ConstraintType": {
        "Budget Constraint": ("budget constraint",),
        "Turnover Constraint": ("turnover constraint",),
        "Tracking Error Constraint": ("tracking error constraint",),
        "Leverage Constraint": ("leverage constraint",),
        "Long-Only Constraint": ("long only", "long-only"),
        "Short-Selling Constraint": ("short selling", "short-selling"),
        "Cardinality Constraint": ("cardinality constraint",),
        "Risk Contribution Constraint": ("risk contribution constraint",),
        "Factor Exposure Constraint": ("factor exposure", "factor constraint"),
    },
    "Estimator": {
        "Historical Estimator": ("historical estimates", "historical estimator"),
        "EWMA": ("ewma",),
        "Ledoit-Wolf": ("ledoit wolf", "ledoit-wolf"),
        "OAS": ("oas",),
        "Shrinkage": ("shrinkage estimator", "shrinkage"),
        "Stepwise Regression": ("stepwise regression",),
        "Principal Components": ("principal components", "pcr"),
    },
    "ReportType": {
        "Performance Report": ("performance report",),
        "Risk Report": ("risk report",),
        "Portfolio Report": ("portfolio report",),
        "Allocation Report": ("allocation report",),
    },
    "PlotType": {
        "Efficient Frontier Plot": ("efficient frontier",),
        "Dendrogram Plot": ("dendrogram",),
        "Network Plot": ("network plot", "asset network"),
        "Risk Contribution Plot": ("risk contribution plot",),
        "Pie Chart": ("pie chart",),
        "Bar Chart": ("bar chart",),
        "Histogram": ("histogram",),
    },
    "Solver": {
        "CVXPY": ("cvxpy",),
        "MOSEK": ("mosek",),
        "ECOS": ("ecos",),
        "OSQP": ("osqp",),
        "SCS": ("scs",),
        "Clarabel": ("clarabel",),
    },
}


def _alias_pattern(alias: str) -> re.Pattern[str]:
    escaped = re.escape(alias.strip().lower())
    escaped = escaped.replace(r"\ ", r"[\s\-_]+")
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", flags=re.IGNORECASE)


DOMAIN_PATTERNS: dict[str, list[tuple[str, re.Pattern[str]]]] = {
    label: [(canonical, _alias_pattern(alias)) for canonical, aliases in concepts.items() for alias in aliases]
    for label, concepts in DOMAIN_ALIASES.items()
}

# Explicit sibling-pair edges for ALTERNATIVE_TO (both directions are emitted)
_ALTERNATIVE_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("CVaR", "VaR", "RiskMeasure"),
    ("EVaR", "CVaR", "RiskMeasure"),
    ("EDaR", "RLDaR", "RiskMeasure"),
    ("Hierarchical Risk Parity", "Hierarchical Equal Risk Contribution", "PortfolioMethod"),
    ("Mean-Variance Optimization", "Minimum Variance", "PortfolioMethod"),
    ("Mean-Variance Optimization", "Maximum Sharpe", "PortfolioMethod"),
)


def emit_taxonomy_edges() -> tuple[list["GraphNode"], list["GraphEdge"]]:
    """Return ontology nodes and edges derived from domain aliases.

    The emitted records model two deterministic structures used throughout the
    graph layer:

    - ``IS_SUBTYPE_OF`` edges from canonical concepts to their category label.
    - bidirectional ``ALTERNATIVE_TO`` edges for selected sibling concepts.

    Returns:
        A tuple of ``(nodes, edges)`` ready to merge into the extracted graph.
    """
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    for category_label, concepts in DOMAIN_ALIASES.items():
        # Category class node (e.g. RiskMeasure → RiskMeasure node)
        nodes.append(GraphNode(label=category_label, name=category_label))
        for canonical_name in concepts:
            nodes.append(GraphNode(label=category_label, name=canonical_name))
            edges.append(
                GraphEdge(
                    source_name=canonical_name,
                    target_name=category_label,
                    relation_type="IS_SUBTYPE_OF",
                    source_label=category_label,
                    target_label=category_label,
                )
            )

    for left_name, right_name, label in _ALTERNATIVE_PAIRS:
        edges.append(
            GraphEdge(
                source_name=left_name,
                target_name=right_name,
                relation_type="ALTERNATIVE_TO",
                source_label=label,
                target_label=label,
            )
        )
        edges.append(
            GraphEdge(
                source_name=right_name,
                target_name=left_name,
                relation_type="ALTERNATIVE_TO",
                source_label=label,
                target_label=label,
            )
        )

    return nodes, edges


@dataclass
class GraphNode:
    """Represent a Neo4j node candidate before deduplication and upsert.

    Attributes:
        label: Neo4j label to merge under, such as ``Chunk`` or ``RiskMeasure``.
        name: Stable logical identifier used as the uniqueness key.
        properties: Flat property payload to merge onto the node.
    """

    label: str
    name: str
    properties: dict[str, str | int] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """Represent a directed Neo4j relationship candidate.

    Attributes:
        source_name: Unique ``name`` property of the source node.
        target_name: Unique ``name`` property of the target node.
        relation_type: Neo4j relationship type, such as ``MENTIONS``.
        source_label: Expected label for the source node match.
        target_label: Expected label for the target node match.
        properties: Flat property payload to merge onto the relationship.
    """

    source_name: str
    target_name: str
    relation_type: str
    source_label: str = "Concept"
    target_label: str = "Concept"
    properties: dict[str, str | int] = field(default_factory=dict)


class LLMExtractProtocol(Protocol):
    """Callable contract for optional chunk-level graph enrichment.

    Implementations may inspect a document chunk and return additional node and
    edge payloads using the same labels and relationship types defined by this
    module.
    """

    def __call__(
        self,
        *,
        content: str,
        source_type: str,
        model_name: str,
    ) -> dict[str, Any]:
        """Extract supplemental graph records from one document chunk.

        Args:
            content: Raw text content of the chunk.
            source_type: Ingestion source classification for the chunk.
            model_name: LLM identifier configured for extraction.

        Returns:
            A dictionary with optional ``nodes`` and ``edges`` lists.
        """
        ...


class GraphBuilder:
    """Build and query the graph-layer representation of repository content.

    The builder owns extraction-to-Neo4j persistence and lightweight graph
    introspection helpers. It deliberately stops at evidence-graph production;
    retrieval orchestration and answer generation live in higher layers.
    """

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        llm_extract: LLMExtractProtocol | None = None,
        llm_model_name: str = "gpt-4o-mini",
    ) -> None:
        """Initialize a graph builder.

        Args:
            neo4j_uri: Neo4j connection URI.
            neo4j_user: Neo4j username.
            neo4j_password: Neo4j password.
            llm_extract: Optional callback that enriches extracted graph records.
            llm_model_name: Model name passed to ``llm_extract`` when enabled.
        """
        self._uri = neo4j_uri
        self._user = neo4j_user
        self._password = neo4j_password
        self._llm_extract = llm_extract
        self._llm_model_name = llm_model_name
        self._driver: Driver | None = None

    def _ensure_driver(self) -> Driver:
        if self._driver is None:
            self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
            self._driver.verify_connectivity()
        return self._driver

    def close(self) -> None:
        """Close the cached Neo4j driver if one has been created."""
        if self._driver is not None:
            self._driver.close()

    def ensure_schema(self, apply_constraints: bool = True) -> None:
        """Create indexes and uniqueness constraints used by graph writes.

        Args:
            apply_constraints: Whether to apply Neo4j constraints and indexes.
                When ``False``, schema creation is skipped entirely.
        """
        if not apply_constraints:
            return

        driver = self._ensure_driver()
        with driver.session() as session:
            for label in NODE_LABELS:
                safe_label = _safe_name(label)
                cypher = (
                    f"CREATE CONSTRAINT {safe_label.lower()}_name_unique IF NOT EXISTS "
                    f"FOR (n:{safe_label}) REQUIRE n.name IS UNIQUE"
                )
                session.run(cypher).consume()

            session.run("CREATE INDEX chunk_source_path IF NOT EXISTS FOR (c:Chunk) ON (c.source_path, c.chunk_index)").consume()

    def build(
        self,
        documents: list[Document],
        drop_existing: bool = False,
        apply_schema: bool = True,
    ) -> None:
        """Extract graph records from documents and persist them to Neo4j.

        Args:
            documents: Chunked ingestion records to transform into graph
                entities and relationships.
            drop_existing: Whether to wipe the existing database before writing.
            apply_schema: Whether to ensure indexes and constraints before upsert.

        Raises:
            RuntimeError: If too many relationships are skipped because their
                endpoints are missing after deduplication.
        """
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        started_at = time.perf_counter()
        total_documents = len(documents)

        logger.info(
            "Graph build started. chunks=%d drop_existing=%s llm_extraction=%s",
            total_documents,
            drop_existing,
            self._llm_extract is not None,
        )

        if total_documents == 0:
            logger.info("No documents were provided to graph build.")

        for index, doc in enumerate(documents, start=1):
            doc_nodes, doc_edges = _extract_entities(
                doc,
                llm_extract=self._llm_extract,
                llm_model_name=self._llm_model_name,
            )
            nodes.extend(doc_nodes)
            edges.extend(doc_edges)

            if index == 1 or index % 25 == 0 or index == total_documents:
                elapsed_seconds = time.perf_counter() - started_at
                progress_percent = (index / total_documents * 100.0) if total_documents else 100.0
                logger.info(
                    "Graph build progress: %d/%d chunks (%.1f%%) elapsed=%.1fs",
                    index,
                    total_documents,
                    progress_percent,
                    elapsed_seconds,
                )

        # Augment with taxonomy (IS_SUBTYPE_OF / ALTERNATIVE_TO) edges.
        taxonomy_nodes, taxonomy_edges = emit_taxonomy_edges()
        nodes.extend(taxonomy_nodes)
        edges.extend(taxonomy_edges)

        unique_nodes = _dedupe_nodes(nodes)
        unique_edges = _dedupe_edges(edges)

        logger.info(
            "Prepared %d nodes (%d unique) and %d edges (%d unique).",
            len(nodes),
            len(unique_nodes),
            len(edges),
            len(unique_edges),
        )

        try:
            driver = self._ensure_driver()
        except (Neo4jError, OSError) as exc:
            logger.warning("Neo4j unavailable; skipping graph writes: %s", exc)
            return

        known_node_names: frozenset[str] = frozenset(node.name for node in unique_nodes)
        with driver.session() as session:
            if drop_existing:
                logger.warning("drop_existing=True – wiping graph.")
                session.run("MATCH (n) DETACH DELETE n").consume()
                logger.info("Existing graph wipe complete.")
            self.ensure_schema(apply_constraints=apply_schema)
            logger.info("Graph schema ready; starting node upsert.")
            _upsert_nodes(session, unique_nodes)
            logger.info("Node upsert complete; starting edge upsert.")
            skipped = _upsert_edges(session, unique_edges, known_node_names=known_node_names)

        ci_threshold = max(1, int(0.05 * len(unique_edges)))
        if skipped > ci_threshold:
            raise RuntimeError(
                f"Edge endpoint integrity failure: {skipped}/{len(unique_edges)} edges skipped "
                f"(exceeds 5% threshold of {ci_threshold})."
            )

        logger.info("Graph write complete: %d nodes, %d edges (%d skipped).", len(unique_nodes), len(unique_edges), skipped)

    def get_stats(self) -> dict[str, int | dict[str, int]]:
        """Return aggregate counts for the current graph.

        Returns:
            A dictionary containing total node and relationship counts plus
            per-label and per-relationship-type breakdowns.
        """
        driver = self._ensure_driver()
        with driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) AS count").single()
            relationship_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()
            label_rows = list(
                session.run("MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS count ORDER BY count DESC")
            )
            relationship_rows = list(
                session.run("MATCH ()-[r]->() RETURN type(r) AS relationship_type, count(*) AS count ORDER BY count DESC")
            )

        node_counts_by_label = {str(row["label"]): int(row["count"]) for row in label_rows}
        relationship_counts_by_type = {str(row["relationship_type"]): int(row["count"]) for row in relationship_rows}

        return {
            "nodes": int(node_count["count"]) if node_count is not None else 0,
            "relationships": int(relationship_count["count"]) if relationship_count is not None else 0,
            "node_counts_by_label": node_counts_by_label,
            "relationship_counts_by_type": relationship_counts_by_type,
        }

    def get_query_subgraph(
        self,
        query: str,
        max_seed_nodes: int = 60,
        max_nodes: int = 300,
    ) -> dict[str, list[dict[str, str | list[str]]]]:
        """Return a bounded one-hop subgraph relevant to a text query.

        The method matches seed nodes by lexical overlap on common textual
        properties, expands one hop, and returns node/edge dictionaries suitable
        for UI visualization.

        Args:
            query: User text used to select lexical seed nodes.
            max_seed_nodes: Maximum number of seed matches before expansion.
            max_nodes: Hard cap on the returned node count after one-hop growth.

        Returns:
            A dictionary with ``nodes`` and ``edges`` lists that can be rendered
            directly by app-layer visualization code.
        """
        terms = _query_terms(query)
        if not terms:
            return {"nodes": [], "edges": []}

        driver = self._ensure_driver()
        with driver.session() as session:
            nodes_result = session.run(
                (
                    "MATCH (n) "
                    "WHERE any(t IN $terms WHERE "
                    "toLower(coalesce(n.name, '')) CONTAINS t OR "
                    "toLower(coalesce(n.content, '')) CONTAINS t OR "
                    "toLower(coalesce(n.relative_path, '')) CONTAINS t) "
                    "WITH collect(DISTINCT n)[0..$max_seed_nodes] AS seeds "
                    "UNWIND seeds AS seed "
                    "OPTIONAL MATCH (seed)-[]-(nbr) "
                    "WITH collect(DISTINCT seed) + collect(DISTINCT nbr) AS raw_nodes "
                    "WITH [n IN raw_nodes WHERE n IS NOT NULL][0..$max_nodes] AS nodes "
                    "RETURN [n IN nodes | {"
                    "id: elementId(n), "
                    "name: coalesce(n.name, ''), "
                    "labels: labels(n), "
                    "source_path: coalesce(n.source_path, coalesce(n.relative_path, ''))"
                    "}] AS nodes"
                ),
                terms=terms,
                max_seed_nodes=max(1, max_seed_nodes),
                max_nodes=max(1, max_nodes),
            ).single()

            nodes = []
            if nodes_result is not None:
                raw_nodes = nodes_result.get("nodes", [])
                if isinstance(raw_nodes, list):
                    nodes = [
                        {
                            "id": str(node.get("id", "")),
                            "name": str(node.get("name", "")),
                            "labels": [str(label) for label in node.get("labels", [])],
                            "source_path": str(node.get("source_path", "")),
                        }
                        for node in raw_nodes
                        if isinstance(node, dict)
                    ]

            if not nodes:
                return {"nodes": [], "edges": []}

            node_ids = [str(node["id"]) for node in nodes if str(node.get("id", ""))]
            edge_rows = list(
                session.run(
                    (
                        "MATCH (a)-[r]->(b) "
                        "WHERE elementId(a) IN $node_ids AND elementId(b) IN $node_ids "
                        "RETURN elementId(a) AS source, elementId(b) AS target, type(r) AS type"
                    ),
                    node_ids=node_ids,
                )
            )

        edges: list[dict[str, str | list[str]]] = [
            {
                "source": str(row["source"]),
                "target": str(row["target"]),
                "type": str(row["type"]),
            }
            for row in edge_rows
        ]
        return {"nodes": nodes, "edges": edges}


def _extract_entities(
    doc: Document,
    llm_extract: LLMExtractProtocol | None = None,
    llm_model_name: str = "gpt-4o-mini",
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Extract Riskfolio-aware graph entities from one chunk."""
    metadata = doc.metadata
    source_type = str(metadata.get("source_type", "python"))
    source_label = _source_label(source_type)
    relative_path = str(metadata.get("relative_path", doc.source_path))
    module_name = str(metadata.get("module_name", ""))
    chunk_kind = str(metadata.get("chunk_kind", "fallback"))
    chunk_id = f"{relative_path}::chunk:{doc.chunk_index}"

    source_name = module_name if source_label == "PythonModule" and module_name else relative_path
    line_start = int(metadata.get("line_start", 1))
    line_end = int(metadata.get("line_end", line_start))

    nodes: list[GraphNode] = [
        GraphNode(
            label=source_label,
            name=source_name,
            properties={
                "source_type": source_type,
                "relative_path": relative_path,
                "module_name": module_name,
            },
        ),
        GraphNode(
            label="Chunk",
            name=chunk_id,
            properties={
                "source_path": doc.source_path,
                "relative_path": relative_path,
                "chunk_index": doc.chunk_index,
                "chunk_kind": chunk_kind,
                "line_start": line_start,
                "line_end": line_end,
                "content": doc.content,
                "source_type": source_type,
                "filename": str(metadata.get("filename", "")),
            },
        ),
    ]

    edges: list[GraphEdge] = [
        GraphEdge(
            source_name=source_name,
            target_name=chunk_id,
            relation_type="HAS_CHUNK",
            source_label=source_label,
            target_label="Chunk",
        )
    ]

    class_names, function_names, function_params = _extract_python_symbols(doc.content)
    if source_type in {"python", "test"}:
        for class_name in class_names:
            nodes.append(GraphNode(label="PythonClass", name=class_name))
            edges.append(
                GraphEdge(
                    source_name=source_name,
                    target_name=class_name,
                    relation_type="DECLARES",
                    source_label=source_label,
                    target_label="PythonClass",
                )
            )
            edges.append(
                GraphEdge(
                    source_name=chunk_id,
                    target_name=class_name,
                    relation_type="MENTIONS",
                    source_label="Chunk",
                    target_label="PythonClass",
                )
            )

        for func_name in function_names:
            nodes.append(GraphNode(label="PythonFunction", name=func_name))
            edges.append(
                GraphEdge(
                    source_name=source_name,
                    target_name=func_name,
                    relation_type="DECLARES",
                    source_label=source_label,
                    target_label="PythonFunction",
                )
            )
            edges.append(
                GraphEdge(
                    source_name=chunk_id,
                    target_name=func_name,
                    relation_type="MENTIONS",
                    source_label="Chunk",
                    target_label="PythonFunction",
                )
            )

            for param in function_params.get(func_name, []):
                parameter_node_name = f"{func_name}:{param}"
                nodes.append(
                    GraphNode(
                        label="Parameter",
                        name=parameter_node_name,
                        properties={"function_name": func_name, "parameter_name": param},
                    )
                )
                edges.append(
                    GraphEdge(
                        source_name=func_name,
                        target_name=parameter_node_name,
                        relation_type="HAS_PARAMETER",
                        source_label="PythonFunction",
                        target_label="Parameter",
                    )
                )

    if source_type == "test":
        for api_name in _extract_test_api_targets(doc.content):
            nodes.append(GraphNode(label="PythonFunction", name=api_name))
            edges.append(
                GraphEdge(
                    source_name=source_name,
                    target_name=api_name,
                    relation_type="VALIDATES",
                    source_label="TestCase",
                    target_label="PythonFunction",
                )
            )

    mentioned_domain_nodes: list[tuple[str, str]] = []
    for label, concept_name in _extract_domain_mentions(doc.content):
        nodes.append(GraphNode(label=label, name=concept_name))
        concept_node_name = _normalize_concept_name(concept_name)
        nodes.append(GraphNode(label="Concept", name=concept_node_name, properties={"canonical": concept_name}))

        mentioned_domain_nodes.append((label, concept_name))

        edges.append(
            GraphEdge(
                source_name=chunk_id,
                target_name=concept_name,
                relation_type="MENTIONS",
                source_label="Chunk",
                target_label=label,
            )
        )
        edges.append(
            GraphEdge(
                source_name=chunk_id,
                target_name=concept_node_name,
                relation_type="MENTIONS",
                source_label="Chunk",
                target_label="Concept",
            )
        )

        source_relation = _concept_source_relation(source_type, label)
        edges.append(
            GraphEdge(
                source_name=source_name,
                target_name=concept_name,
                relation_type=source_relation,
                source_label=source_label,
                target_label=label,
            )
        )

    for (left_label, left_name), (right_label, right_name) in combinations(sorted(set(mentioned_domain_nodes)), 2):
        edges.append(
            GraphEdge(
                source_name=left_name,
                target_name=right_name,
                relation_type="RELATED_TO",
                source_label=left_label,
                target_label=right_label,
            )
        )

    llm_nodes, llm_edges = _extract_entities_with_llm(
        doc=doc,
        source_name=source_name,
        source_label=source_label,
        chunk_id=chunk_id,
        llm_extract=llm_extract,
        llm_model_name=llm_model_name,
    )
    nodes.extend(llm_nodes)
    edges.extend(llm_edges)

    return nodes, edges


def _extract_entities_with_llm(
    doc: Document,
    source_name: str,
    source_label: str,
    chunk_id: str,
    llm_extract: LLMExtractProtocol | None,
    llm_model_name: str,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    if llm_extract is None:
        return [], []

    source_type = str(doc.metadata.get("source_type", "python"))
    try:
        payload = llm_extract(
            content=doc.content,
            source_type=source_type,
            model_name=llm_model_name,
        )
    except Exception as exc:
        logger.warning("LLM extraction failed for chunk %s: %s", chunk_id, exc)
        return [], []

    if not isinstance(payload, dict):
        return [], []

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    for item in payload.get("nodes", []):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        name = str(item.get("name", "")).strip()
        if not label or not name or label not in NODE_LABELS:
            continue

        properties: dict[str, str | int] = {}
        raw_properties = item.get("properties", {})
        if isinstance(raw_properties, dict):
            for key, value in raw_properties.items():
                if not isinstance(key, str):
                    continue
                if isinstance(value, str | int):
                    properties[key] = value

        nodes.append(GraphNode(label=label, name=name, properties=properties))
        if label != "Chunk":
            edges.append(
                GraphEdge(
                    source_name=chunk_id,
                    target_name=name,
                    relation_type="MENTIONS",
                    source_label="Chunk",
                    target_label=label,
                    properties={"origin": "llm"},
                )
            )

    for item in payload.get("edges", []):
        if not isinstance(item, dict):
            continue
        relation_type = str(item.get("relation_type", "")).strip()
        source_node_name = str(item.get("source_name", "")).strip()
        target_node_name = str(item.get("target_name", "")).strip()
        source_node_label = str(item.get("source_label", "")).strip() or "Concept"
        target_node_label = str(item.get("target_label", "")).strip() or "Concept"

        if (
            not relation_type
            or relation_type not in RELATIONSHIP_TYPES
            or not source_node_name
            or not target_node_name
            or source_node_label not in NODE_LABELS
            or target_node_label not in NODE_LABELS
        ):
            continue

        edges.append(
            GraphEdge(
                source_name=source_node_name,
                target_name=target_node_name,
                relation_type=relation_type,
                source_label=source_node_label,
                target_label=target_node_label,
                properties={"origin": "llm"},
            )
        )

    if nodes:
        anchored_nodes = [node for node in nodes if node.label in {"PythonModule", "DocPage", "ExampleNotebook", "TestCase"}]
        for node in anchored_nodes:
            if node.name == source_name and node.label == source_label:
                continue
            edges.append(
                GraphEdge(
                    source_name=source_name,
                    target_name=node.name,
                    relation_type="RELATED_TO",
                    source_label=source_label,
                    target_label=node.label,
                    properties={"origin": "llm"},
                )
            )

    return nodes, edges


def _extract_python_symbols(content: str) -> tuple[list[str], list[str], dict[str, list[str]]]:
    class_matches = re.findall(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\b", content, flags=re.M)
    func_matches = re.findall(
        r"^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        content,
        flags=re.M,
    )

    function_params: dict[str, list[str]] = {}
    for func_name, params_blob in re.findall(
        r"^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*:",
        content,
        flags=re.M,
    ):
        params: list[str] = []
        for raw in params_blob.split(","):
            token = raw.strip()
            if not token:
                continue
            token = token.split("=")[0].strip().lstrip("*")
            if token in {"self", "cls", "/"}:
                continue
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token):
                params.append(token)
        if params:
            function_params[func_name] = params

    return sorted(set(class_matches)), sorted(set(func_matches)), function_params


def _extract_test_api_targets(content: str) -> list[str]:
    candidates = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", content)
    blocked = {
        "assert",
        "print",
        "len",
        "range",
        "list",
        "dict",
        "set",
        "tuple",
        "int",
        "float",
        "str",
        "bool",
        "enumerate",
        "zip",
    }

    valid: list[str] = []
    for name in candidates:
        if name.startswith("test_"):
            continue
        if name in blocked or keyword.iskeyword(name):
            continue
        valid.append(name)
    return sorted(set(valid))


def _extract_domain_mentions(content: str) -> list[tuple[str, str]]:
    mentions: list[tuple[str, str]] = []
    for label, patterns in DOMAIN_PATTERNS.items():
        for canonical, pattern in patterns:
            if pattern.search(content):
                mentions.append((label, canonical))
    return sorted(set(mentions))


def _source_label(source_type: str) -> str:
    if source_type == "docs":
        return "DocPage"
    if source_type == "example":
        return "ExampleNotebook"
    if source_type == "test":
        return "TestCase"
    return "PythonModule"


def _concept_source_relation(source_type: str, label: str) -> str:
    if label == "RiskMeasure":
        return "SUPPORTS_RISK_MEASURE"
    if label == "ConstraintType":
        return "SUPPORTS_CONSTRAINT"
    if label == "Estimator":
        return "USES_ESTIMATOR"
    if label == "ReportType":
        return "PRODUCES_REPORT"
    if label == "Solver":
        return "USES"

    if source_type == "docs":
        return "DESCRIBES"
    if source_type == "example":
        return "DEMONSTRATES"
    if source_type == "test":
        return "VALIDATES"
    return "IMPLEMENTS"


def _normalize_concept_name(name: str) -> str:
    normalized = re.sub(r"\s+", " ", name.strip().lower())
    return normalized


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "", value) or "Entity"


def _query_terms(query: str) -> list[str]:
    terms = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", query.lower())
    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        deduped.append(term)
    return deduped[:12]


def _dedupe_nodes(nodes: list[GraphNode]) -> list[GraphNode]:
    deduped: dict[tuple[str, str], GraphNode] = {}
    for node in nodes:
        key = (node.label, node.name)
        if key not in deduped:
            deduped[key] = GraphNode(label=node.label, name=node.name, properties=dict(node.properties))
            continue
        deduped[key].properties.update(node.properties)
    return list(deduped.values())


def _dedupe_edges(edges: list[GraphEdge]) -> list[GraphEdge]:
    deduped: dict[tuple[str, str, str, str, str], GraphEdge] = {}
    for edge in edges:
        key = (
            edge.source_label,
            edge.source_name,
            edge.relation_type,
            edge.target_label,
            edge.target_name,
        )
        if key not in deduped:
            deduped[key] = GraphEdge(
                source_name=edge.source_name,
                target_name=edge.target_name,
                relation_type=edge.relation_type,
                source_label=edge.source_label,
                target_label=edge.target_label,
                properties=dict(edge.properties),
            )
            continue
        deduped[key].properties.update(edge.properties)
    return list(deduped.values())


def _batched_rows(rows: list[dict[str, Any]], batch_size: int = WRITE_BATCH_SIZE) -> list[list[dict[str, Any]]]:
    return [rows[index : index + batch_size] for index in range(0, len(rows), batch_size)]


def _upsert_nodes(session, nodes: list[GraphNode]) -> None:
    if not nodes:
        return

    by_label: dict[str, list[GraphNode]] = {}
    for node in nodes:
        by_label.setdefault(node.label, []).append(node)

    for label, labeled_nodes in by_label.items():
        safe_label = _safe_name(label)
        cypher = f"UNWIND $rows AS row MERGE (n:{safe_label} {{name: row.name}}) SET n += row.properties"
        rows = [{"name": n.name, "properties": n.properties} for n in labeled_nodes]
        processed = 0
        for batch in _batched_rows(rows):
            session.run(cypher, rows=batch).consume()
            processed += len(batch)
        logger.info("Node upserted: label=%s count=%d", label, processed)


def _upsert_edges(
    session,
    edges: list[GraphEdge],
    known_node_names: frozenset[str] | None = None,
) -> int:
    """Upsert edges into Neo4j. Returns the count of skipped (bad-endpoint) edges."""
    if not edges:
        return 0

    skipped = 0
    grouped_rows: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for edge in edges:
        if known_node_names is not None:
            if edge.source_name not in known_node_names or edge.target_name not in known_node_names:
                logger.warning(
                    "Skipping edge (%s)-[%s]->(%s): one or both endpoints absent from node set.",
                    edge.source_name,
                    edge.relation_type,
                    edge.target_name,
                )
                skipped += 1
                continue

        group_key = (
            _safe_name(edge.source_label),
            _safe_name(edge.target_label),
            _safe_name(edge.relation_type),
        )
        grouped_rows.setdefault(group_key, []).append(
            {
                "source_name": edge.source_name,
                "target_name": edge.target_name,
                "properties": edge.properties,
            }
        )

    total_edges = sum(len(rows) for rows in grouped_rows.values())
    processed = 0
    for (safe_source_label, safe_target_label, safe_relation), rows in grouped_rows.items():
        cypher = (
            "UNWIND $rows AS row "
            f"MATCH (s:{safe_source_label} {{name: row.source_name}}) "
            f"MATCH (t:{safe_target_label} {{name: row.target_name}}) "
            f"MERGE (s)-[r:{safe_relation}]->(t) "
            "SET r += row.properties"
        )
        for batch in _batched_rows(rows):
            session.run(cypher, rows=batch).consume()
            processed += len(batch)
            if processed == total_edges or processed % 1000 == 0:
                logger.info("Edge upsert progress: %d/%d", processed, total_edges)
    return skipped
