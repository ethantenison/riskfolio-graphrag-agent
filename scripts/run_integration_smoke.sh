#!/usr/bin/env bash
set -euo pipefail

# End-to-end smoke profile: Neo4j + vector backend + ingestion + graph + query + eval.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
export NEO4J_USER="${NEO4J_USER:-neo4j}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-password}"
export VECTOR_STORE_BACKEND="${VECTOR_STORE_BACKEND:-chroma}"
export RETRIEVAL_MODE="${RETRIEVAL_MODE:-hybrid_rerank}"
export EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-hash}"

echo "[integration] starting Neo4j + Chroma services"
docker compose -f docker-compose.integration.yml up -d

echo "[integration] waiting for services"
sleep 8

echo "[integration] running ingestion"
poetry run riskfolio-agent ingest --source-dir src

echo "[integration] building graph (bounded sample)"
poetry run riskfolio-agent build-graph --drop-existing --source-dir src --max-chunks 10

echo "[integration] running one query via agent workflow"
poetry run python -c "from riskfolio_graphrag_agent.config.settings import Settings; from riskfolio_graphrag_agent.retrieval.embeddings import resolve_embedding_provider; from riskfolio_graphrag_agent.retrieval.retriever import HybridRetriever; from riskfolio_graphrag_agent.agent.workflow import AgentWorkflow; s=Settings(); p=resolve_embedding_provider(provider_name=s.embedding_provider, embedding_dim=s.embedding_dim, openai_api_key=s.openai_api_key, openai_embedding_model=s.embedding_model, openai_base_url=s.openai_base_url, openai_timeout_seconds=s.openai_timeout_seconds); r=HybridRetriever(neo4j_uri=s.neo4j_uri, neo4j_user=s.neo4j_user, neo4j_password=s.neo4j_password, top_k=3, vector_store_backend=s.vector_store_backend, chroma_persist_dir=s.chroma_persist_dir, embedding_provider=p.provider, retrieval_mode=s.retrieval_mode); w=AgentWorkflow(retriever=r, model_name=s.openai_model, llm_generate=None); st=w.run('What is Hierarchical Risk Parity?'); print(st.answer); r.close()"

echo "[integration] running eval + gates"
poetry run riskfolio-agent eval --output eval_results.json
poetry run riskfolio-agent eval-gate --report eval_results.json

echo "[integration] generating semantic + observability artifacts"
poetry run python scripts/benchmark_retrieval_ablation.py
poetry run python scripts/report_observability.py

echo "[integration] smoke workflow complete"
echo "Expected outputs: eval_results.json, benchmarks/retrieval_ablation_results.json, artifacts/observability/sli_report.json"
