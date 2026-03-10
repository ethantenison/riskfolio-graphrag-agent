#!/usr/bin/env python3
"""Export RDF/OWL, execute SPARQL examples, and run SHACL-like validation."""

from __future__ import annotations

import json
from pathlib import Path

from riskfolio_graphrag_agent.config.settings import Settings
from riskfolio_graphrag_agent.graph.semantic_interop import (
    export_rdf_owl_from_neo4j,
    run_basic_sparql_queries,
    shacl_like_validate,
)


def _load_graph_snapshot(settings: Settings) -> tuple[list[dict[str, str | list[str]]], list[dict[str, str]]]:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
    try:
        with driver.session() as session:
            nodes = [
                {"name": str(row["name"]), "labels": [str(v) for v in row["labels"]]}
                for row in session.run("MATCH (n) WHERE n.name IS NOT NULL RETURN n.name AS name, labels(n) AS labels LIMIT 300")
            ]
            edges = [
                {
                    "source": str(row["source"]),
                    "target": str(row["target"]),
                    "relation": str(row["relation"]),
                }
                for row in session.run(
                    (
                        "MATCH (a)-[r]->(b) "
                        "WHERE a.name IS NOT NULL AND b.name IS NOT NULL "
                        "RETURN a.name AS source, b.name AS target, type(r) AS relation LIMIT 600"
                    )
                )
            ]
    finally:
        driver.close()

    return nodes, edges


def main() -> int:
    settings = Settings()
    artifacts_dir = Path("artifacts/semantic")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    rdf_path = artifacts_dir / "graph_export.ttl"
    export_info = export_rdf_owl_from_neo4j(
        neo4j_uri=settings.neo4j_uri,
        neo4j_user=settings.neo4j_user,
        neo4j_password=settings.neo4j_password,
        output_path=rdf_path,
    )

    sparql_results = run_basic_sparql_queries(rdf_path)
    nodes, edges = _load_graph_snapshot(settings)
    validation = shacl_like_validate(nodes=nodes, edges=edges, output_path=artifacts_dir / "shacl_like_report.json")

    summary = {
        "export": export_info,
        "sparql": sparql_results,
        "validation": validation,
    }
    summary_path = artifacts_dir / "semantic_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"wrote {rdf_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
