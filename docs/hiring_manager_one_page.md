# Hiring Manager One-Page Summary

## What this project demonstrates

This repository is a production-credible GraphRAG portfolio artifact for an enterprise Knowledge Graph / Agentic AI role:
- End-to-end architecture across ingestion, Neo4j graph construction, retrieval, agent workflow, API/UI, evaluation, and governance.
- Hybrid retrieval supporting dense, sparse, graph-only, and hybrid rerank modes.
- Enterprise graph interoperability with RDF/OWL export, SPARQL examples, and SHACL-like validation.
- Entity-resolution pipeline with deterministic canonicalization, optional model-assist hook, and ER precision/recall/F1.
- Guarded NL-to-Cypher with read-only enforcement, allowlisted templates, audit logs, and human escalation.
- Evaluation scorecard expanded to faithfulness, grounding, multi-hop, ER metrics, link prediction proxies, latency, and cost.
- Observability hardening via configurable OTLP tracing and SLI/SLO reporting including drift/freshness signals.

## Why it maps to Dell’s posting

- Semantic architecture and enterprise KG operation: Neo4j + RDF/OWL + Cypher/SPARQL path.
- Robust retrieval and reasoning: hybrid vector/symbolic/graph retrieval with explicit ablations.
- Agentic safety and governance: guardrails, auditability, regression gates, and trend artifacts.
- Evaluation leadership: clear measurable scorecard and CI enforcement.
- Platform readiness: dockerized integration workflow and reproducible smoke run.

## Evidence links

- Capability matrix: `docs/capability_matrix.md`
- Architecture map: `docs/architecture_module_map.md`
- Metrics delta: `docs/before_after_metrics.md`
- Integration workflow: `scripts/run_integration_smoke.sh`
