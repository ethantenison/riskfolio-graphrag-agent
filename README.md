---
title: GraphRAG Riskfolio
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.0"
app_file: app.py
pinned: false
---

# riskfolio-graphrag-agent

> **Explainable GraphRAG + Agentic AI demo** over the [Riskfolio-Lib](https://riskfolio-lib.readthedocs.io/) codebase and documentation.

[![CI](https://github.com/ethantenison/riskfolio-graphrag-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/ethantenison/riskfolio-graphrag-agent/actions/workflows/ci.yml)

---

## Goals

This is a **portfolio project** that demonstrates:

- **Knowledge graph construction** – entities (functions, classes, parameters, concepts) extracted from the Riskfolio-Lib source and docs are stored in Neo4j.
- **Hybrid retrieval (GraphRAG)** – queries combine vector similarity search with graph-neighbourhood traversal for richer, more precise context.
- **Agentic workflow** – a LangGraph-based multi-step agent plans, retrieves, reasons, and verifies before answering.
- **Explainability & provenance** – every answer is accompanied by citations linking back to the original source files and graph nodes.
- **Evaluation** – a built-in evaluation suite measures context recall, precision, faithfulness, and answer relevance.

> **Disclaimer:** This project is a technical demo only. It does not provide financial advice.

---

## Architecture

```
┌─────────────┐    ┌────────────────┐    ┌─────────────────┐
│  CLI / API  │───▶│  Agent (plan,  │───▶│  LLM (OpenAI /  │
│  (Typer /   │    │  retrieve,     │    │  compatible)    │
│  FastAPI)   │    │  reason,       │    └─────────────────┘
└─────────────┘    │  verify)       │
                   └───────┬────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
   ┌─────────────────┐       ┌──────────────────┐
   │  Vector Store   │       │  Neo4j Knowledge │
   │  (Chroma /      │       │  Graph           │
   │  Qdrant)        │       │  (entities,      │
   └────────┬────────┘       │   relations)     │
            │                └────────┬─────────┘
            └─────────────────────────┘
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


### Observability & Tracing

This project is instrumented with OpenTelemetry and LangSmith for full agent workflow tracing and evaluation:

- Agent workflow, retrieval, and graph operations are traced with OpenTelemetry spans.
- LangSmith tracing decorates agentic workflow for step-level inspection.
- FastAPI exposes `/trace` endpoint for trace status and demo.
- Evaluation suite includes faithfulness, grounding, precision/recall, and multi-hop metrics.
- All code is modular and production-ready for enterprise KG/RAG/agentic AI deployment.

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

This demonstrates advanced governance, explainability, and observability—matching enterprise requirements for roles like Dell’s Knowledge Graph / RAG Agentic AI Expert.

### Module Map

| Package | Responsibility |
|---|---|
| `config/` | Pydantic-Settings based configuration from env/`.env` |
| `ingestion/` | Walk source dirs, chunk files, produce `Document` objects |
| `graph/` | Extract entities from chunks, upsert nodes/edges to Neo4j |
| `retrieval/` | Hybrid vector + graph search returning cited `RetrievalResult` |
| `agent/` | LangGraph workflow: plan → retrieve → reason → verify |
| `eval/` | Evaluation harness (context recall, faithfulness, relevance) |
| `app/` | FastAPI server exposing `/health`, `/query`, `/graph/stats` |

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
- [x] Eval: RAGAS-style metrics
- [x] Eval: CI evaluation regression gate
- [x] Observability: LangSmith / OpenTelemetry tracing

---

## License

[MIT](LICENSE)
