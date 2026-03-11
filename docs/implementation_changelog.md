# Implementation Summary and File-by-File Change Log

## Retrieval and embeddings
- `src/riskfolio_graphrag_agent/retrieval/embeddings.py`
  - Added provider abstraction (`EmbeddingProvider`) with OpenAI and hash fallback implementations.
  - Added explicit fallback metadata (`ProviderResolution`).
- `src/riskfolio_graphrag_agent/retrieval/retriever.py`
  - Added retrieval ablation modes: dense, sparse, graph, hybrid rerank.
  - Integrated embedding providers into Chroma upsert/query path.
  - Added sparse and graph seed Cypher retrieval and hybrid score merge logic.
- `scripts/benchmark_retrieval_ablation.py`
  - Added fixed-eval ablation benchmark outputting JSON and markdown.
- `benchmarks/retrieval_ablation_results.json`
- `benchmarks/retrieval_ablation_results.md`

## Enterprise KG semantics
- `src/riskfolio_graphrag_agent/graph/semantic_interop.py`
  - Added RDF/OWL export from Neo4j/records.
  - Added SPARQL query runner.
  - Added SHACL-like structural validation report.
- `scripts/run_semantic_checks.py`
- `benchmarks/sparql_examples.rq`

## Entity resolution
- `src/riskfolio_graphrag_agent/er/pipeline.py`
  - Added deterministic canonicalization, optional model-assist merge, pair prediction.
  - Added ER precision/recall/F1 and audit artifact output.
- `src/riskfolio_graphrag_agent/cli.py`
  - Added `er-eval` command and integrated ER metrics in `eval` command.

## Guarded NL-to-Cypher
- `src/riskfolio_graphrag_agent/graph/nl2cypher_guard.py`
  - Added allowlisted query templates, read-only enforcement, and escalation conditions.
  - Added audit log appender.
- `src/riskfolio_graphrag_agent/app/server.py`
  - Added `POST /graph/nl2cypher` endpoint with audited decisions.

## Evaluation and governance
- `src/riskfolio_graphrag_agent/eval/evaluator.py`
  - Expanded scorecard: grounding, multi-hop, ER metrics, link prediction proxy metrics, latency, cost.
- `src/riskfolio_graphrag_agent/eval/regression_gate.py`
  - Added expanded gate thresholds and trend artifact writer.
- `eval_results.json`
  - Added expanded scorecard fields.

## Observability
- `src/riskfolio_graphrag_agent/observability/reporting.py`
  - Added SLI/SLO report with drift and freshness checks.
- `scripts/report_observability.py`
- `src/riskfolio_graphrag_agent/app/server.py`
  - Made OTLP tracing endpoint configurable via settings.
  - Added span attributes for tenant/request/model/cost/workflow stage.
- `src/riskfolio_graphrag_agent/agent/workflow.py`
  - Added per-stage trace attributes.

## Integration profile and CI
- `docker-compose.integration.yml`
  - Added Neo4j + Chroma integration services.
- `scripts/run_integration_smoke.sh`
  - Added single-command integration workflow.
- `.github/workflows/ci.yml`
  - Added benchmark and observability reporting steps.

## Tests
- Added/updated tests:
  - `tests/test_embeddings.py`
  - `tests/test_semantic_interop.py`
  - `tests/test_entity_resolution.py`
  - `tests/test_nl2cypher_guard.py`
  - `tests/test_observability.py`
  - `tests/test_integration_artifacts.py`
  - Updated existing tests in retrieval/app/eval gate/config compatibility areas.
