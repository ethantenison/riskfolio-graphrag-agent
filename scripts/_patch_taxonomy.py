#!/usr/bin/env python3
"""One-shot script: patch the live Neo4j graph with taxonomy edges."""

from __future__ import annotations

from collections import defaultdict

from neo4j import GraphDatabase

from riskfolio_graphrag_agent.graph.builder import _safe_name, emit_taxonomy_edges

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
taxonomy_nodes, taxonomy_edges = emit_taxonomy_edges()

with driver.session() as session:
    by_label: dict = defaultdict(list)
    for n in taxonomy_nodes:
        by_label[n.label].append(n)
    for label, nodes in by_label.items():
        safe = _safe_name(label)
        session.run(
            f"UNWIND $rows AS row MERGE (n:{safe} {{name: row.name}})",
            rows=[{"name": n.name} for n in nodes],
        )
    print(f"upserted {len(taxonomy_nodes)} taxonomy nodes")

    node_names = {n.name for n in taxonomy_nodes}
    skipped = 0
    for e in taxonomy_edges:
        if e.source_name not in node_names or e.target_name not in node_names:
            skipped += 1
            continue
        ss = _safe_name(e.source_label)
        ts = _safe_name(e.target_label)
        rel = _safe_name(e.relation_type)
        session.run(
            f"MATCH (s:{ss} {{name: $src}}) MATCH (t:{ts} {{name: $tgt}}) MERGE (s)-[r:{rel}]->(t)",
            src=e.source_name,
            tgt=e.target_name,
        )
    print(f"upserted {len(taxonomy_edges) - skipped} taxonomy edges (skipped {skipped})")

driver.close()
