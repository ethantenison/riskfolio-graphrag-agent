"""RDF/OWL export, SPARQL examples, and SHACL-like graph validation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase
from rdflib import Graph, Literal, Namespace
from rdflib.namespace import OWL, RDF, RDFS

from riskfolio_graphrag_agent.graph.builder import NODE_LABELS, RELATIONSHIP_TYPES

logger = logging.getLogger(__name__)

RF = Namespace("https://example.com/riskfolio/kg#")


def export_rdf_owl_from_records(
    *,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    output_path: str | Path,
) -> dict[str, int | str]:
    graph = build_rdf_graph(nodes=nodes, edges=edges)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(destination=str(target), format="turtle")

    return {
        "output_path": str(target),
        "triples": len(graph),
        "nodes": len(nodes),
        "edges": len(edges),
    }


def export_rdf_owl_from_neo4j(
    *,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    output_path: str | Path,
    node_limit: int = 300,
    edge_limit: int = 600,
) -> dict[str, int | str]:
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    try:
        with driver.session() as session:
            node_rows = list(
                session.run(
                    ("MATCH (n) WHERE n.name IS NOT NULL RETURN n.name AS name, labels(n) AS labels LIMIT $limit"),
                    limit=max(1, node_limit),
                )
            )
            edge_rows = list(
                session.run(
                    (
                        "MATCH (a)-[r]->(b) "
                        "WHERE a.name IS NOT NULL AND b.name IS NOT NULL "
                        "RETURN a.name AS source, b.name AS target, type(r) AS relation "
                        "LIMIT $limit"
                    ),
                    limit=max(1, edge_limit),
                )
            )
    finally:
        driver.close()

    nodes = [{"name": str(row["name"]), "labels": [str(v) for v in row["labels"]]} for row in node_rows]
    edges = [
        {
            "source": str(row["source"]),
            "target": str(row["target"]),
            "relation": str(row["relation"]),
        }
        for row in edge_rows
    ]

    return export_rdf_owl_from_records(nodes=nodes, edges=edges, output_path=output_path)


def build_rdf_graph(*, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> Graph:
    graph = Graph()
    graph.bind("rf", RF)
    graph.bind("owl", OWL)

    for label in NODE_LABELS:
        graph.add((RF[label], RDF.type, OWL.Class))
        graph.add((RF[label], RDFS.label, Literal(label)))

    for relation in RELATIONSHIP_TYPES:
        graph.add((RF[relation], RDF.type, OWL.ObjectProperty))
        graph.add((RF[relation], RDFS.label, Literal(relation)))

    for node in nodes:
        name = str(node.get("name", "")).strip()
        if not name:
            continue
        labels = [str(label) for label in node.get("labels", [])]
        subject = _node_uri(name)
        graph.add((subject, RDFS.label, Literal(name)))
        for label in labels:
            graph.add((subject, RDF.type, RF[label]))

    for edge in edges:
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        relation = str(edge.get("relation", "")).strip()
        if not source or not target or not relation:
            continue
        graph.add((_node_uri(source), RF[relation], _node_uri(target)))

    return graph


def run_basic_sparql_queries(rdf_path: str | Path) -> dict[str, list[dict[str, str]]]:
    graph = Graph()
    graph.parse(str(rdf_path), format="turtle")

    node_count_query = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?name WHERE {
      ?s rdfs:label ?name .
    }
    LIMIT 5
    """

    rel_query = """
    PREFIX rf: <https://example.com/riskfolio/kg#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?source ?target WHERE {
      ?s rf:MENTIONS ?t .
      ?s rdfs:label ?source .
      ?t rdfs:label ?target .
    }
    LIMIT 5
    """

    sample_nodes = [{"name": str(row["name"])} for row in graph.query(node_count_query)]
    sample_mentions = [{"source": str(row["source"]), "target": str(row["target"])} for row in graph.query(rel_query)]

    return {
        "sample_nodes": sample_nodes,
        "sample_mentions": sample_mentions,
    }


def shacl_like_validate(
    *,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    node_names = {str(node.get("name", "")).strip() for node in nodes if str(node.get("name", "")).strip()}

    checks = {
        "node_has_name": {
            "pass": sum(1 for node in nodes if str(node.get("name", "")).strip()),
            "fail": sum(1 for node in nodes if not str(node.get("name", "")).strip()),
        },
        "allowed_relation_type": {
            "pass": sum(1 for edge in edges if str(edge.get("relation", "")) in RELATIONSHIP_TYPES),
            "fail": sum(1 for edge in edges if str(edge.get("relation", "")) not in RELATIONSHIP_TYPES),
        },
        "edge_endpoints_exist": {
            "pass": sum(
                1
                for edge in edges
                if str(edge.get("source", "")).strip() in node_names and str(edge.get("target", "")).strip() in node_names
            ),
            "fail": sum(
                1
                for edge in edges
                if str(edge.get("source", "")).strip() not in node_names or str(edge.get("target", "")).strip() not in node_names
            ),
        },
    }

    total_pass = sum(section["pass"] for section in checks.values())
    total_fail = sum(section["fail"] for section in checks.values())
    report = {
        "status": "pass" if total_fail == 0 else "fail",
        "checks": checks,
        "pass_count": total_pass,
        "fail_count": total_fail,
        "pass_rate": round(total_pass / max(1, total_pass + total_fail), 4),
    }

    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2))

    return report


def _node_uri(name: str):
    cleaned = "_".join(name.strip().split())
    return RF[f"node/{cleaned}"]
