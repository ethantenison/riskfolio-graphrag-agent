---
title: GraphRAG Riskfolio
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.23.1"
app_file: app.py
pinned: false
---

# riskfolio-graphrag-agent

> **GraphRAG / hybrid retrieval demo and evaluation scaffold** over the [Riskfolio-Lib](https://riskfolio-lib.readthedocs.io/) codebase and documentation.

[![CI](https://github.com/ethantenison/riskfolio-graphrag-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/ethantenison/riskfolio-graphrag-agent/actions/workflows/ci.yml)

---

## Goals

This is a **portfolio project** that demonstrates:

- **Knowledge graph construction** – entities (functions, classes, parameters, concepts) extracted from the Riskfolio-Lib source and docs are stored in Neo4j.
- **Hybrid retrieval (GraphRAG)** – queries combine vector similarity search with graph-neighbourhood traversal for richer, more precise context.
- **Agentic workflow** – a LangGraph-based multi-step agent plans, retrieves, reasons, and verifies before answering.
- **Explainability & provenance** – every answer is accompanied by citations linking back to the original source files and graph nodes.
- **Evaluation** – a built-in evaluation suite uses heuristic overlap/support metrics for regression tracking and small benchmark comparisons.

> **Disclaimer:** This project is a technical demo only. It does not provide financial advice.

---

## Role Fit (Knowledge Graph / GraphRAG / Agentic AI)

This repository is intentionally structured as evidence for senior roles involving knowledge graphs, GraphRAG, semantic modeling, and agentic AI systems. The mapping below is written to be reusable across similar roles rather than tailored to one company or posting.

### Capability-to-Evidence Mapping

| Capability area | Evidence in this portfolio | Broader experience alignment |
|---|---|---|
| End-to-end LLM, RAG/GraphRAG, and agentic architecture with observability, governance, and cost awareness | End-to-end flow across [src/riskfolio_graphrag_agent/ingestion/loader.py](src/riskfolio_graphrag_agent/ingestion/loader.py), [src/riskfolio_graphrag_agent/graph/builder.py](src/riskfolio_graphrag_agent/graph/builder.py), [src/riskfolio_graphrag_agent/retrieval/retriever.py](src/riskfolio_graphrag_agent/retrieval/retriever.py), [src/riskfolio_graphrag_agent/agent/workflow.py](src/riskfolio_graphrag_agent/agent/workflow.py), and [src/riskfolio_graphrag_agent/app/server.py](src/riskfolio_graphrag_agent/app/server.py); observability and SLI/SLO reporting in [src/riskfolio_graphrag_agent/observability/reporting.py](src/riskfolio_graphrag_agent/observability/reporting.py) | Aligns with prior delivery of production AI systems that required deployment discipline, monitoring, and cost-aware design |
| Taxonomy, ontology, and semantic modeling | Ontology-aware entity and relationship extraction plus curated domain aliases in [src/riskfolio_graphrag_agent/graph/builder.py](src/riskfolio_graphrag_agent/graph/builder.py); semantic export and validation helpers in [src/riskfolio_graphrag_agent/graph/semantic_interop.py](src/riskfolio_graphrag_agent/graph/semantic_interop.py) | Aligns with prior work on typed content hierarchies, canonical concepts, and retrieval-aware information design |
| Build and operate knowledge graphs with Neo4j, RDF/OWL, Cypher, and SPARQL-oriented workflows | Neo4j graph construction and stats in [src/riskfolio_graphrag_agent/graph/builder.py](src/riskfolio_graphrag_agent/graph/builder.py); guarded graph querying in [src/riskfolio_graphrag_agent/graph/nl2cypher_guard.py](src/riskfolio_graphrag_agent/graph/nl2cypher_guard.py); RDF/OWL-style export and SPARQL examples in [src/riskfolio_graphrag_agent/graph/semantic_interop.py](src/riskfolio_graphrag_agent/graph/semantic_interop.py) and [benchmarks/sparql_examples.rq](benchmarks/sparql_examples.rq) | Demonstrates practical graph engineering with semantic-web interoperability rather than graph storage alone |
| Extraction and linking pipelines with disambiguation, deduplication, canonicalization, and QA | Entity-resolution pipeline in [src/riskfolio_graphrag_agent/er/pipeline.py](src/riskfolio_graphrag_agent/er/pipeline.py); supporting audit artifacts in [artifacts/er/er_audit.json](artifacts/er/er_audit.json) | Aligns with prior extraction and normalization work where content quality and canonical linking mattered operationally |
| Production LLM and agentic workflows using frameworks such as LangGraph and related orchestration stacks | Plan-retrieve-reason-verify orchestration in [src/riskfolio_graphrag_agent/agent/workflow.py](src/riskfolio_graphrag_agent/agent/workflow.py); request orchestration in [src/riskfolio_graphrag_agent/app/server.py](src/riskfolio_graphrag_agent/app/server.py) | Aligns with broader production experience using LLM-backed workflows and orchestration patterns beyond a single demo |
| Safe tool use, tracing, and human-in-the-loop style controls | Safe NL-to-Cypher guardrails and audit logging in [src/riskfolio_graphrag_agent/graph/nl2cypher_guard.py](src/riskfolio_graphrag_agent/graph/nl2cypher_guard.py); tracing and request lifecycle visibility in [src/riskfolio_graphrag_agent/app/server.py](src/riskfolio_graphrag_agent/app/server.py) | Shows a safety-first approach to letting models interact with structured systems |
| Advanced retrieval blending vector, symbolic, and KG retrieval | Dense, sparse, graph, and hybrid retrieval in [src/riskfolio_graphrag_agent/retrieval/retriever.py](src/riskfolio_graphrag_agent/retrieval/retriever.py); adaptive routing in [src/riskfolio_graphrag_agent/retrieval/router.py](src/riskfolio_graphrag_agent/retrieval/router.py) | Demonstrates hybrid retrieval design and ontology-guided search patterns rather than plain vector search |
| Evaluation and observability for RAG/GraphRAG and graph systems | Evaluation logic in [src/riskfolio_graphrag_agent/eval/evaluator.py](src/riskfolio_graphrag_agent/eval/evaluator.py) and [src/riskfolio_graphrag_agent/eval/regression_gate.py](src/riskfolio_graphrag_agent/eval/regression_gate.py); output artifacts in [eval_results.json](eval_results.json) and [artifacts/observability/sli_report.json](artifacts/observability/sli_report.json) | Covers grounding, faithfulness, latency, drift, and operational quality in a way that supports governance discussions |
| Strong Python and AI/ML engineering | Typed Python modules, configuration, testing, CI, and deployment-facing app surfaces across [src/riskfolio_graphrag_agent](src/riskfolio_graphrag_agent) and [tests](tests) | Complements broader ML and production engineering experience in optimization, experimentation, and platform delivery |
| Graph ML, reranking, and multi-hop reasoning patterns | Embedding-backed retrieval, hybrid reranking, graph expansion, and multi-hop style evidence gathering in [src/riskfolio_graphrag_agent/retrieval/retriever.py](src/riskfolio_graphrag_agent/retrieval/retriever.py) | Strong on graph-aware retrieval and reasoning; lighter on learned graph neural models specifically |
| Communication, leadership, and stakeholder-facing system design | This repository emphasizes explainability, architectural clarity, documentation, and evidence traceability across [README.md](README.md), [docs/architecture_module_map.md](docs/architecture_module_map.md), and the app surfaces | Best paired with resume and interview examples showing mentoring, cross-functional delivery, and product influence |

### What this repository demonstrates especially well

- **Semantic architecture and KG design** through ontology-aware extraction, canonical graph representation, and graph-backed retrieval.
- **Graph + GenAI integration** through GraphRAG retrieval, agentic orchestration, and grounded answer generation.
- **Governance and explainability** through citations, route visibility, NL-to-Cypher safety controls, and auditability.
- **Evaluation and observability** through measurable quality gates, tracing, latency/cost reporting, and drift-aware operational artifacts.

### Notes on evidence scope

- This repository provides **public, inspectable implementation evidence** for the core technical areas above.
- Some capabilities, especially **leadership, stakeholder influence, and proprietary production deployments**, are more fully demonstrated in resume and interview materials than in a public code repository.
- For interview review, this README plus [docs/architecture_module_map.md](docs/architecture_module_map.md) and the [artifacts](artifacts) directory provide concrete evidence of architecture choices, quality measurement, and execution discipline.

---

## Architecture

```
┌──────────────────┐   ┌────────────────┐   ┌─────────────────┐
│  Gradio UI /     │──▶│  Query Router  │──▶│  Agent (plan,   │
│  CLI / FastAPI   │   │  (dense/sparse │   │  retrieve,      │
└──────────────────┘   │   /graph/      │   │  reason,        │
                       │   hybrid)      │   │  verify/retry)  │
                       └────────────────┘   └────────┬────────┘
                                                     │
                                    ┌────────────────┼────────────────┐
                                    ▼                ▼                ▼
                         ┌──────────────┐  ┌──────────────┐  ┌────────────┐
                         │  Dense:      │  │  Sparse:     │  │  Graph:    │
                         │  Embedding   │  │  Neo4j       │  │  Neo4j     │
                         │  Provider +  │  │  Cypher      │  │  1-hop     │
                         │  Vector Store│  │  lexical     │  │  expansion │
                         └──────┬───────┘  └──────┬───────┘  └─────┬──────┘
                                │                  │                │
                                └──────────────────┴────────────────┘
                                                   │
                                    ┌──────────────▼──────────────┐
                                    │  LLM (OpenAI / compatible)  │
                                    │  reason step → answer +     │
                                    │  citations                  │
                                    └─────────────────────────────┘
                                    ▲
                                    │ embed + upsert
                             ┌──────┴──────┐
                             │  Ingestion  │
                             │  (chunker,  │
                             │   extractor)│
                             └──────┬──────┘
                                    │
                             ┌──────┴──────┐
                             │ Riskfolio-  │
                             │ Lib source  │
                             │ + docs      │
                             └─────────────┘
```

The **Query Router** (`retrieval/router.py`) inspects each question with
rule-based intent detection and lightweight embedding similarity to pick the
best retrieval mode before the agent workflow begins.  The **Agent Workflow**
(`agent/workflow.py`) is a four-node LangGraph state machine: *plan* decomposes
the question into sub-questions; *retrieve* calls `HybridRetriever`; *reason*
generates the answer via the LLM; *verify* checks grounding and retries the
reason step up to twice if needed.  The Gradio UI additionally issues a
`GraphBuilder.get_query_subgraph()` call after the workflow to populate the
interactive graph visualisation panel.


### Observability & Tracing

This project is instrumented with OpenTelemetry and LangSmith for full agent workflow tracing and evaluation:

- Agent workflow, retrieval, and graph operations are traced with OpenTelemetry spans.
- LangSmith tracing decorates agentic workflow for step-level inspection.
- FastAPI exposes `/trace` endpoint for trace status and demo.
- Evaluation suite includes faithfulness, grounding, precision/recall, and multi-hop metrics.
- Retrieval includes a lexical Neo4j fallback for deterministic local runs; this repository should be treated as a demo scaffold rather than a production-ready deployment package.

#### OpenTelemetry + Jaeger Setup

To view traces in Jaeger:

1. Start Jaeger with OTLP enabled:
       ```bash
       docker run -d --name jaeger \
         -e COLLECTOR_OTLP_ENABLED=true \
         -p 4317:4317 \
         -p 16686:16686 \
         jaegertracing/all-in-one:latest
       ```
       - Port 4317 is for OTLP gRPC (traces from FastAPI).
       - Port 16686 is for the Jaeger web UI (http://localhost:16686).

2. Restart your FastAPI app:
       ```bash
       poetry run riskfolio-agent serve --host 127.0.0.1 --port 8000
       ```

3. Submit queries (e.g. with curl):
       ```bash
       curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" -d '{"question":"HRP in Riskfolio?","top_k":3}'
       ```

4. Open Jaeger UI at http://localhost:16686 and search for traces from "riskfolio-graphrag-agent".

You’ll see spans for each request, including agent workflow steps (plan, retrieve, reason, verify).

#### LangSmith Tracing

To use LangSmith, set your API key:
```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your-key-here
export LANGCHAIN_PROJECT=RiskfolioGraphRAG
```
Restart your app and view traces in your LangSmith dashboard.

This demonstrates observability and governance patterns that are useful for portfolio review, while still leaving substantial validation and hardening work for any production setting.

### Module Map

| Package | Responsibility |
|---|---|
| `config/` | Pydantic-Settings based configuration from env/`.env` |
| `ingestion/` | Walk source dirs, chunk files, produce `Document` objects |
| `graph/` | Extract entities from chunks, upsert nodes/edges to Neo4j |
| `retrieval/retriever.py` | Hybrid vector + graph search returning cited `RetrievalResult` |
| `retrieval/router.py` | Adaptive query routing: selects `dense`, `sparse`, `graph`, or `hybrid_rerank` per question |
| `agent/` | LangGraph workflow: plan → retrieve → reason → verify (with self-correction retry) |
| `eval/` | Evaluation harness (context recall, faithfulness, relevance) |
| `app/server.py` | FastAPI endpoints (`/health`, `/query`, `/graph/stats`) with OTel tracing |
| `app/gradio_ui.py` | Gradio chat UI, interactive graph visualisation panel, and insight displays |

---

## Local Setup

### Prerequisites

- Python 3.13+
- [Poetry](https://python-poetry.org/docs/#installation)
- Docker & Docker Compose (for Neo4j)

### 1 – Clone and install

```bash
git clone https://github.com/ethantenison/riskfolio-graphrag-agent.git
cd riskfolio-graphrag-agent
poetry install
```

### 2 – Configure environment

```bash
#cp .env.example .env
# Edit .env and fill in OPENAI_API_KEY and RISKFOLIO_SOURCE_DIR
```

### 3 – Start Neo4j

```bash
docker compose up -d
# Neo4j Browser: http://localhost:7474
```

### 4 – Ingest source material

```bash
# Uses RISKFOLIO_SOURCE_DIR from .env (recommended)
poetry run riskfolio-agent ingest

# Or pass an explicit path override
poetry run riskfolio-agent ingest --source-dir /Users/et/Desktop/Data_Projects/Riskfolio-Lib
```

### 5 – Build knowledge graph

```bash
# First build
poetry run riskfolio-agent build-graph

# After changes
poetry run riskfolio-agent build-graph --drop-existing

# Target a specific window of chunks (skip first 100, then process 2)
poetry run riskfolio-agent build-graph --drop-existing --chunk-offset 100 --max-chunks 2
```

### 6 – Ask a question

```bash
# FastAPI API
poetry run riskfolio-agent serve --host 127.0.0.1 --port 8000
# curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" -d '{"question":"HRP in Riskfolio?","top_k":3}'

# Gradio chat interface + graph visualisation
poetry run riskfolio-agent gradio --host 127.0.0.1 --port 7860
```

### API Docs

Once the server is running, you can explore the API in:

- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc
- OpenAPI JSON: http://127.0.0.1:8000/openapi.json

Current endpoints:

- `GET /health`
- `GET /graph/stats`
- `POST /query`

### 7 – Run evaluation

```bash
poetry run riskfolio-agent eval --output eval_results.json

# Legacy deterministic profile (for comparison)
poetry run riskfolio-agent eval --metric-profile heuristic --output eval_results.json
```

---

## Development

```bash
# Run tests
poetry run pytest

# Lint
poetry run ruff check src tests

# Format
poetry run ruff format src tests
```

---

## Roadmap

- [x] Project scaffold (Poetry, src layout, CLI, Docker Compose, CI)
- [x] Ingestion: AST-based Python chunker with docstring/signature extraction
- [x] Ingestion: RST/Markdown section splitter
- [x] Graph: LLM-assisted entity & relationship extraction (OpenAI-compatible JSON + heuristic fallback)
- [x] Graph: Ontology design for Riskfolio concepts (Portfolio, Asset, Metric, Method)
- [x] Retrieval: ChromaDB vector store integration
- [x] Retrieval: Neo4j graph traversal queries (Cypher)
- [x] Retrieval: Hybrid re-ranking
- [x] Agent: LangGraph workflow with tool use, model-backed generation, and self-correction
- [x] App: FastAPI endpoints + OpenAPI docs
- [x] App: Gradio chat interface with graph visualisation
- [x] Eval: CI evaluation regression gate
- [x] Observability: LangSmith / OpenTelemetry tracing

---

## Known Limitations

For a more comprehensive benchmark set, see [benchmarks/eval_samples_v1.json](benchmarks/eval_samples_v1.json). For more details on current limitations, see [docs/limitations.md](docs/limitations.md).

---

## License

[MIT](LICENSE)
