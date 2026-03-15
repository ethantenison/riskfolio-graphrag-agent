"""Bridge the Neo4j graph into semantic-web style artifacts and checks.

This module sits in the graph layer alongside the Neo4j builder, but its scope
is interoperability rather than extraction. It projects graph records into an
RDF/OWL view, runs a small set of example SPARQL queries, and produces a
lightweight SHACL-like validation report.

Inputs are node and edge dictionaries or Neo4j connection settings. Outputs are
Turtle files, ``rdflib.Graph`` objects, example query result payloads, and JSON
validation summaries.

Key implementation decisions:
- ontology classes and properties are declared from the same label and
    relationship registries used by the builder so exports stay aligned with the
    primary graph schema;
- a stable project namespace is used for generated URIs to keep artifacts
    reproducible across runs;
- validation is intentionally lightweight and artifact-oriented rather than a
    full SHACL engine.

This module does not own Neo4j extraction, answer generation, or UI rendering.

Example:
        summary = export_rdf_owl_from_records(nodes=nodes, edges=edges, output_path="artifacts/semantic/graph.ttl")
        report = shacl_like_validate(nodes=nodes, edges=edges)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD

from riskfolio_graphrag_agent.graph.builder import DOMAIN_ALIASES, NODE_LABELS, RELATIONSHIP_TYPES

logger = logging.getLogger(__name__)

# Project-stable ontology namespace (v1)
RF = Namespace("https://riskfolio-graphrag.io/ontology/v1#")

# External ontology namespace prefixes
FIBO_SEC = Namespace("https://spec.edmcouncil.org/fibo/ontology/SEC/")
FIBO_FBC = Namespace("https://spec.edmcouncil.org/fibo/ontology/FBC/")
STATO = Namespace("http://purl.obolibrary.org/obo/STATO_")
QUDT = Namespace("http://qudt.org/schema/qudt/")
PROV = Namespace("http://www.w3.org/ns/prov#")

RELATION_PROPERTY_AXIOMS: tuple[tuple[str, URIRef | None, URIRef | None], ...] = (
    ("MENTIONS", OWL.Thing, OWL.Thing),
    ("IS_SUBTYPE_OF", OWL.Thing, OWL.Thing),
    ("ALTERNATIVE_TO", OWL.Thing, OWL.Thing),
    ("REQUIRES", OWL.Thing, OWL.Thing),
    ("PARAMETERIZED_BY", OWL.Thing, OWL.Thing),
    ("SUPPORTS_RISK_MEASURE", RF["PortfolioMethod"], RF["RiskMeasure"]),
    ("USES_ESTIMATOR", RF["PortfolioMethod"], RF["Estimator"]),
    ("HAS_PARAMETER", OWL.Thing, RF["Parameter"]),
    ("BENCHMARKED_ON", RF["BacktestScenario"], RF["BenchmarkIndex"]),
    ("CALIBRATED_ON", RF["Estimator"], RF["AssetClass"]),
    ("PRECEDES", OWL.Thing, OWL.Thing),
    ("HAS_CONSTRAINT", RF["PortfolioMethod"], RF["ConstraintType"]),
    ("VALIDATED_AGAINST", OWL.Thing, OWL.Thing),
    ("RELATED_TO", OWL.Thing, OWL.Thing),
    ("DESCRIBES", OWL.Thing, OWL.Thing),
    ("DEMONSTRATES", OWL.Thing, OWL.Thing),
    ("VALIDATES", OWL.Thing, OWL.Thing),
    ("IMPLEMENTS", OWL.Thing, OWL.Thing),
)

_RELATION_AXIOM_LOOKUP: dict[str, tuple[URIRef | None, URIRef | None]] = {
    relation: (domain_cls, range_cls) for relation, domain_cls, range_cls in RELATION_PROPERTY_AXIOMS
}


def export_rdf_owl_from_records(
    *,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    output_path: str | Path,
) -> dict[str, int | str]:
    """Serialize in-memory graph records to Turtle.

    Args:
        nodes: Node dictionaries containing at least ``name`` and optionally
            ``labels``.
        edges: Edge dictionaries containing ``source``, ``target``, and
            ``relation`` fields.
        output_path: Destination path for the Turtle file.

    Returns:
        A summary containing the output path and record counts used to build the
        RDF graph.
    """
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
    """Query Neo4j for a bounded graph sample and export it as Turtle.

    Args:
        neo4j_uri: Neo4j connection URI.
        neo4j_user: Neo4j username.
        neo4j_password: Neo4j password.
        output_path: Destination path for the Turtle file.
        node_limit: Maximum number of named nodes to export.
        edge_limit: Maximum number of relationships to export.

    Returns:
        The same artifact summary returned by ``export_rdf_owl_from_records``.
    """
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
    """Construct an RDF/OWL projection of the application graph.

    Args:
        nodes: Node dictionaries with ``name`` and optional ``labels`` fields.
        edges: Edge dictionaries with ``source``, ``target``, and ``relation``.

    Returns:
        An ``rdflib.Graph`` containing ontology declarations, instance triples,
        and exported relationships.
    """
    graph = Graph()
    graph.bind("rf", RF)
    graph.bind("owl", OWL)
    graph.bind("rdfs", RDFS)
    graph.bind("xsd", XSD)
    graph.bind("fibo-sec", FIBO_SEC)
    graph.bind("fibo-fbc", FIBO_FBC)
    graph.bind("stato", STATO)
    graph.bind("qudt", QUDT)
    graph.bind("prov", PROV)

    # --- OWL class declarations for every node label ---
    for label in NODE_LABELS:
        graph.add((RF[label], RDF.type, OWL.Class))
        graph.add((RF[label], RDFS.label, Literal(label)))

    # --- OWL object-property declarations with domain / range axioms ---
    for relation in RELATIONSHIP_TYPES:
        graph.add((RF[relation], RDF.type, OWL.ObjectProperty))
        graph.add((RF[relation], RDFS.label, Literal(relation)))
    for prop_name, domain_cls, range_cls in RELATION_PROPERTY_AXIOMS:
        if domain_cls is not None:
            graph.add((RF[prop_name], RDFS.domain, domain_cls))
        if range_cls is not None:
            graph.add((RF[prop_name], RDFS.range, range_cls))

    # --- OWL datatype properties ---
    _datatype_props = (
        "hasConfidenceLevel",
        "hasTargetReturn",
        "hasWeightUpperBound",
        "hasWeightLowerBound",
        "hasRiskBudgetFraction",
    )
    for dp_name in _datatype_props:
        graph.add((RF[dp_name], RDF.type, OWL.DatatypeProperty))
        graph.add((RF[dp_name], RDFS.label, Literal(dp_name)))
        graph.add((RF[dp_name], RDFS.range, XSD.float))

    # --- owl:equivalentClass alignments to FIBO / STATO ---
    graph.add((RF["PortfolioMethod"], OWL.equivalentClass, FIBO_SEC["InvestmentStrategy"]))
    graph.add((RF["RiskMeasure"], OWL.equivalentClass, FIBO_FBC["RiskMeasure"]))
    graph.add((RF["Estimator"], OWL.equivalentClass, STATO["0000118"]))  # stato:StatisticalEstimator

    # --- rdfs:subClassOf hierarchy from DOMAIN_ALIASES ---
    for category_label, concepts in DOMAIN_ALIASES.items():
        category_uri = RF[_slug(category_label)]
        graph.add((category_uri, RDF.type, OWL.Class))
        graph.add((category_uri, RDFS.label, Literal(category_label)))
        for canonical_name in concepts:
            concept_uri = RF[_slug(canonical_name)]
            graph.add((concept_uri, RDF.type, OWL.Class))
            graph.add((concept_uri, RDFS.label, Literal(canonical_name)))
            graph.add((concept_uri, RDFS.subClassOf, category_uri))

    # --- instance nodes ---
    for node in nodes:
        name = str(node.get("name", "")).strip()
        if not name:
            continue
        labels = [str(label) for label in node.get("labels", [])]
        subject = _node_uri(name)
        graph.add((subject, RDFS.label, Literal(name)))
        for label in labels:
            graph.add((subject, RDF.type, RF[label]))

    # --- edges ---
    for edge in edges:
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        relation = str(edge.get("relation", "")).strip()
        if not source or not target or not relation:
            continue
        graph.add((_node_uri(source), RF[relation], _node_uri(target)))

    return graph


def run_basic_sparql_queries(rdf_path: str | Path) -> dict[str, list[dict[str, str]]]:
    """Run sample SPARQL queries over an exported Turtle artifact.

    Args:
        rdf_path: Path to a Turtle file produced by this module.

    Returns:
        A dictionary of sample result sets keyed by query purpose. The payload is
        designed for smoke tests, demos, and artifact inspection rather than for
        general analytics.
    """
    graph = Graph()
    graph.parse(str(rdf_path), format="turtle")

    _RF_PREFIX = "PREFIX rf: <https://riskfolio-graphrag.io/ontology/v1#>"
    _RDFS_PREFIX = "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>"

    node_count_query = f"""
    {_RDFS_PREFIX}
    SELECT ?name WHERE {{
      ?s rdfs:label ?name .
    }}
    LIMIT 5
    """

    subclass_query = f"""
    {_RF_PREFIX}
    {_RDFS_PREFIX}
    SELECT ?child ?parent WHERE {{
      ?s rdfs:subClassOf ?p .
      ?s rdfs:label ?child .
      ?p rdfs:label ?parent .
    }}
    LIMIT 10
    """

    alternative_query = f"""
    {_RF_PREFIX}
    {_RDFS_PREFIX}
    SELECT ?source ?target WHERE {{
      ?s rf:ALTERNATIVE_TO ?t .
      ?s rdfs:label ?source .
      ?t rdfs:label ?target .
    }}
    LIMIT 10
    """

    supports_risk_query = f"""
    {_RF_PREFIX}
    {_RDFS_PREFIX}
    SELECT ?method ?measure WHERE {{
      ?m rf:SUPPORTS_RISK_MEASURE ?r .
      ?m rdfs:label ?method .
      ?r rdfs:label ?measure .
    }}
    LIMIT 10
    """

    uses_estimator_query = f"""
    {_RF_PREFIX}
    {_RDFS_PREFIX}
    SELECT ?method ?estimator WHERE {{
      ?m rf:USES_ESTIMATOR ?e .
      ?m rdfs:label ?method .
      ?e rdfs:label ?estimator .
    }}
    LIMIT 10
    """

    sample_nodes = [{"name": str(row["name"])} for row in graph.query(node_count_query)]
    sample_subclass = [{"child": str(row["child"]), "parent": str(row["parent"])} for row in graph.query(subclass_query)]
    sample_alternative = [{"source": str(row["source"]), "target": str(row["target"])} for row in graph.query(alternative_query)]
    sample_supports = [{"method": str(row["method"]), "measure": str(row["measure"])} for row in graph.query(supports_risk_query)]
    sample_estimator = [
        {"method": str(row["method"]), "estimator": str(row["estimator"])} for row in graph.query(uses_estimator_query)
    ]

    return {
        "sample_nodes": sample_nodes,
        "sample_mentions": sample_subclass,  # kept for backward-compat key name
        "sample_subclass": sample_subclass,
        "sample_alternative_to": sample_alternative,
        "sample_supports_risk_measure": sample_supports,
        "sample_uses_estimator": sample_estimator,
    }


def describe_relationship_semantics(relation: str) -> dict[str, str]:
    """Return compact RDF/OWL metadata for a relationship type.

    Args:
        relation: Graph relationship type, such as ``SUPPORTS_RISK_MEASURE``.

    Returns:
        A semantic description containing a compact RDF predicate name, the
        OWL property type, and any known domain/range constraints.
    """
    normalized = relation.strip()
    domain_cls, range_cls = _RELATION_AXIOM_LOOKUP.get(normalized, (OWL.Thing, OWL.Thing))
    return {
        "relation": normalized,
        "predicate": f"rf:{normalized}",
        "predicate_uri": str(RF[normalized]),
        "owl_type": "owl:ObjectProperty",
        "domain": _compact_uri(domain_cls),
        "range": _compact_uri(range_cls),
    }


def shacl_like_validate(
    *,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run lightweight structural validation checks over graph records.

    Args:
        nodes: Node dictionaries that may be exported or validated.
        edges: Edge dictionaries that may be exported or validated.
        output_path: Optional path for writing the JSON validation report.

    Returns:
        A report containing per-check pass and fail counts plus aggregate status
        metrics.
    """
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


def _slug(name: str) -> str:
    """Convert a human-readable name to a URI-safe slug (spaces/hyphens → underscores)."""
    import re as _re

    return _re.sub(r"[^A-Za-z0-9_]", "_", name.strip())


def _node_uri(name: str):
    cleaned = "_".join(name.strip().split())
    return RF[f"node/{cleaned}"]


def _compact_uri(value: URIRef | None) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.startswith(str(RF)):
        return f"rf:{text.removeprefix(str(RF))}"
    if text.startswith(str(OWL)):
        return f"owl:{text.removeprefix(str(OWL))}"
    if text.startswith(str(RDFS)):
        return f"rdfs:{text.removeprefix(str(RDFS))}"
    return text
