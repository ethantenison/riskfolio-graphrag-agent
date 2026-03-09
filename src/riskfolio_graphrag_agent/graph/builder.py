"""Riskfolio-aware knowledge graph builder for Neo4j."""

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
        "CVaR": ("cvar", "conditional value at risk"),
        "VaR": ("value at risk", "var"),
        "Semi Deviation": ("semi deviation", "semi standard deviation", "semi-std"),
        "Semi Variance": ("semi variance", "semivariance"),
        "MAD": ("mean absolute deviation", "mad"),
        "Ulcer Index": ("ulcer index",),
        "EVaR": ("entropic value at risk", "evar"),
        "EDaR": ("entropic drawdown at risk", "edar"),
        "RLVaR": ("relativistic value at risk", "rlvar"),
        "RLDaR": ("relativistic drawdown at risk", "rldar"),
        "Tail Gini": ("tail gini",),
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
    label: [
        (canonical, _alias_pattern(alias))
        for canonical, aliases in concepts.items()
        for alias in aliases
    ]
    for label, concepts in DOMAIN_ALIASES.items()
}


@dataclass
class GraphNode:
    """A graph node for Neo4j upsert."""

    label: str
    name: str
    properties: dict[str, str | int] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """A directed graph edge for Neo4j upsert."""

    source_name: str
    target_name: str
    relation_type: str
    source_label: str = "Concept"
    target_label: str = "Concept"
    properties: dict[str, str | int] = field(default_factory=dict)


class LLMExtractProtocol(Protocol):
    def __call__(
        self,
        *,
        content: str,
        source_type: str,
        model_name: str,
    ) -> dict[str, Any]:
        ...


class GraphBuilder:
    """Coordinates Riskfolio-aware extraction and Neo4j writes."""

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        llm_extract: LLMExtractProtocol | None = None,
        llm_model_name: str = "gpt-4o-mini",
    ) -> None:
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
        if self._driver is not None:
            self._driver.close()

    def ensure_schema(self, apply_constraints: bool = True) -> None:
        """Create optional indexes/constraints for idempotent graph writes."""
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
                session.run(cypher)

            session.run(
                "CREATE INDEX chunk_source_path IF NOT EXISTS "
                "FOR (c:Chunk) ON (c.source_path, c.chunk_index)"
            )

    def build(
        self,
        documents: list[Document],
        drop_existing: bool = False,
        apply_schema: bool = True,
    ) -> None:
        """Extract entities from chunks and upsert a Riskfolio graph."""
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

        with driver.session() as session:
            if drop_existing:
                logger.warning("drop_existing=True – wiping graph.")
                session.run("MATCH (n) DETACH DELETE n")
            self.ensure_schema(apply_constraints=apply_schema)
            _upsert_nodes(session, unique_nodes)
            _upsert_edges(session, unique_edges)

        logger.info(
            "Graph write complete: %d nodes, %d edges.", len(unique_nodes), len(unique_edges)
        )

    def get_stats(self) -> dict[str, int | dict[str, int]]:
        """Return graph counts by label and relationship type."""
        driver = self._ensure_driver()
        with driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) AS count").single()
            relationship_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()
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

        node_counts_by_label = {str(row["label"]): int(row["count"]) for row in label_rows}
        relationship_counts_by_type = {
            str(row["relationship_type"]): int(row["count"]) for row in relationship_rows
        }

        return {
            "nodes": int(node_count["count"]) if node_count is not None else 0,
            "relationships": int(relationship_count["count"])
            if relationship_count is not None
            else 0,
            "node_counts_by_label": node_counts_by_label,
            "relationship_counts_by_type": relationship_counts_by_type,
        }


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
        nodes.append(
            GraphNode(
                label="Concept", name=concept_node_name, properties={"canonical": concept_name}
            )
        )

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

    for (left_label, left_name), (right_label, right_name) in combinations(
        sorted(set(mentioned_domain_nodes)), 2
    ):
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
                if isinstance(value, (str, int)):
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
        anchored_nodes = [
            node
            for node in nodes
            if node.label in {"PythonModule", "DocPage", "ExampleNotebook", "TestCase"}
        ]
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


def _dedupe_nodes(nodes: list[GraphNode]) -> list[GraphNode]:
    deduped: dict[tuple[str, str], GraphNode] = {}
    for node in nodes:
        key = (node.label, node.name)
        if key not in deduped:
            deduped[key] = GraphNode(
                label=node.label, name=node.name, properties=dict(node.properties)
            )
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


def _upsert_nodes(session, nodes: list[GraphNode]) -> None:
    if not nodes:
        return

    by_label: dict[str, list[GraphNode]] = {}
    for node in nodes:
        by_label.setdefault(node.label, []).append(node)

    for label, labeled_nodes in by_label.items():
        safe_label = _safe_name(label)
        cypher = (
            f"UNWIND $rows AS row MERGE (n:{safe_label} {{name: row.name}}) SET n += row.properties"
        )
        rows = [{"name": n.name, "properties": n.properties} for n in labeled_nodes]
        session.run(cypher, rows=rows)


def _upsert_edges(session, edges: list[GraphEdge]) -> None:
    if not edges:
        return

    for edge in edges:
        safe_source_label = _safe_name(edge.source_label)
        safe_target_label = _safe_name(edge.target_label)
        safe_relation = _safe_name(edge.relation_type)
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
