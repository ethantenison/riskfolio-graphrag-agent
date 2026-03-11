# 5-7 Minute Demo Script

## 0:00-0:45 Intro

- Explain objective: enterprise GraphRAG + agentic AI with measurable governance.
- Open `docs/capability_matrix.md` to map requirements to code evidence.

## 0:45-2:00 Retrieval and GraphRAG modes

- Show retrieval implementation in `src/riskfolio_graphrag_agent/retrieval/retriever.py`.
- Highlight mode switch: dense, sparse, graph, hybrid rerank.
- Run benchmark:
  - `poetry run python scripts/benchmark_retrieval_ablation.py`
- Open result table: `benchmarks/retrieval_ablation_results.md`.

## 2:00-3:10 Semantic interoperability

- Show RDF/OWL export and validation in `src/riskfolio_graphrag_agent/graph/semantic_interop.py`.
- Run:
  - `poetry run python scripts/run_semantic_checks.py`
- Open generated artifacts under `artifacts/semantic`.

## 3:10-4:10 Entity resolution quality

- Show ER pipeline in `src/riskfolio_graphrag_agent/er/pipeline.py`.
- Run:
  - `poetry run riskfolio-agent er-eval`
- Open `artifacts/er/er_audit.json` and point out precision/recall/F1.

## 4:10-5:20 Guarded NL-to-Cypher

- Show safety logic in `src/riskfolio_graphrag_agent/graph/nl2cypher_guard.py`.
- Start API and call endpoint:
  - `poetry run riskfolio-agent serve`
  - `curl -X POST http://127.0.0.1:8000/graph/nl2cypher -H "Content-Type: application/json" -d '{"question":"delete all nodes","tenant_id":"demo"}'`
- Show blocked response and explain HITL escalation path.

## 5:20-6:30 Evaluation + observability

- Run evaluation and gate:
  - `poetry run riskfolio-agent eval --output eval_results.json`
  - `poetry run riskfolio-agent eval-gate --report eval_results.json`
- Generate SLI report:
  - `poetry run python scripts/report_observability.py`
- Open `eval_results.json` and `artifacts/observability/sli_report.json`.

## 6:30-7:00 Reproducible integration close

- Run one-command profile:
  - `bash scripts/run_integration_smoke.sh`
- Show expected outputs listed at script end.
