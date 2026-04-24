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

> Provenance-first knowledge graph induction and GraphRAG over the Riskfolio-Lib codebase and documentation.

[![CI](https://github.com/ethantenison/riskfolio-graphrag-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/ethantenison/riskfolio-graphrag-agent/actions/workflows/ci.yml)

## Overview

This repository implements a staged knowledge graph induction system with:

- open chunk-level extraction,
- mention and assertion persistence,
- explicit canonicalization decisions,
- reviewable schema induction,
- controlled Neo4j materialization for retrieval,
- RDF/OWL/SKOS/PROV export over stabilized graph layers,
- graph-quality evaluation in addition to answer-quality evaluation.

> Disclaimer: this repository is a technical system-design and engineering artifact. It does not provide financial advice.

## Main Components

| Area | Responsibility |
|---|---|
| [src/riskfolio_graphrag_agent/ingestion/loader.py](src/riskfolio_graphrag_agent/ingestion/loader.py) | Chunk source code, docs, examples, and tests into provenance-rich `Document` records |
| [src/riskfolio_graphrag_agent/extraction/pipeline.py](src/riskfolio_graphrag_agent/extraction/pipeline.py) | Open extraction into mentions, candidate entities, candidate assertions, and candidate events |
| [src/riskfolio_graphrag_agent/canonicalization/pipeline.py](src/riskfolio_graphrag_agent/canonicalization/pipeline.py) | Canonicalize open extraction outputs into reviewable entity, predicate, and event-type clusters |
| [src/riskfolio_graphrag_agent/schema_induction/pipeline.py](src/riskfolio_graphrag_agent/schema_induction/pipeline.py) | Aggregate open semantic guesses into schema candidates and induced ontology elements |
| [src/riskfolio_graphrag_agent/graph_materialization/pipeline.py](src/riskfolio_graphrag_agent/graph_materialization/pipeline.py) | Materialize a narrower Neo4j retrieval graph with constraints and representative Cypher queries |
| [src/riskfolio_graphrag_agent/semantic_export/pipeline.py](src/riskfolio_graphrag_agent/semantic_export/pipeline.py) | Export ontology and instance/provenance Turtle views using OWL, SKOS, and PROV-O |
| [src/riskfolio_graphrag_agent/evaluation/graph_quality.py](src/riskfolio_graphrag_agent/evaluation/graph_quality.py) | Score graph-quality metrics such as compression, promotion yield, and schema support |
| [src/riskfolio_graphrag_agent/kg_pipeline.py](src/riskfolio_graphrag_agent/kg_pipeline.py) | Orchestrate the end-to-end KG pipeline and artifact generation |
| [src/riskfolio_graphrag_agent/retrieval/retriever.py](src/riskfolio_graphrag_agent/retrieval/retriever.py) | Retrieval orchestration supporting four modes: dense (embedding), sparse (lexical), graph (assertion-aware), and hybrid_rerank (merged with graph context) |
| [src/riskfolio_graphrag_agent/retrieval/router.py](src/riskfolio_graphrag_agent/retrieval/router.py) | Query router to select appropriate retrieval mode |
| [src/riskfolio_graphrag_agent/agent/workflow.py](src/riskfolio_graphrag_agent/agent/workflow.py) | Plan-retrieve-reason-verify orchestration for answer generation |

## Architecture

```text
Source Files / Docs
        │
        ▼
Ingestion → Chunk Records
        │
        ▼
Open Extraction
  - MentionRecord
  - CandidateEntityRecord
  - CandidateAssertionRecord
  - CandidateEventRecord
        │
        ▼
Canonicalization
  - CanonicalEntity
  - CanonicalPredicate
  - CanonicalEventType
  - CanonicalizationDecision
        │
        ▼
Schema Induction
  - SchemaInductionCandidate
  - OntologyClass
  - OntologyProperty
  - ConceptScheme
        │
        ├──► Semantic Export
        │     - ontology.ttl
        │     - instances.ttl
        │
        ▼
Graph Materialization
  - GraphWritePlan
  - Neo4j constraints / writes
  - retrieval-oriented Cypher
        │
        ▼
Promoted Retrieval Graph
```

## Retrieval Methods

The retrieval system supports four modes to serve different query characteristics:

| Mode | Strategy | Best For | Candidate Pool |
|------|----------|----------|-----------------|
| **Dense** | Embedding-based semantic similarity with query expansion | Broad semantic queries; paraphrasing | 2×top_k |
| **Sparse** | Direct lexical token matching on Neo4j chunks | Domain terminology; acronyms; exact phrases | 1×top_k |
| **Graph** | Shallow assertion-aware traversal: entity seed, ontology-class expansion, assertion-bridged peers, keyword backfill | Relationship and entity-centric queries | 3×top_k |
| **Hybrid Rerank** | Merged dense and sparse results with graph-contextualized evidence boosting (default) | General-purpose balanced precision/recall | 4×top_k |

All modes (except sparse) apply optional reranking boosts based on:
- **Entity signal**: Richness of related entity context (~log scale, max 0.07–0.11 boost)
- **Neighbour signal**: Graph connectivity (~log scale, max 0.03–0.07 boost)
- **Coverage signal**: Fraction of query tokens present in text (0–0.09 boost)

For detailed mode descriptions, performance profiles, formulas, and troubleshooting, see [docs/retrieval_methods.md](docs/retrieval_methods.md).

## Commands

```bash
poetry run riskfolio-agent ingest --source-dir /path/to/Riskfolio-Lib
poetry run riskfolio-agent kg-run --source-dir /path/to/Riskfolio-Lib --artifact-dir artifacts/kg
```

The `kg-run` command writes reviewable artifacts including:

- `artifacts/kg/extractions.json`
- `artifacts/kg/canonicalization.json`
- `artifacts/kg/schema_candidates.json`
- `artifacts/kg/schema_review.md`
- `artifacts/kg/materialized_graph.json`
- `artifacts/kg/graph_quality.json`
- `artifacts/kg/semantic/ontology.ttl`
- `artifacts/kg/semantic/instances.ttl`

To also write the promoted retrieval graph into Neo4j:

```bash
poetry run riskfolio-agent kg-run --source-dir /path/to/Riskfolio-Lib --artifact-dir artifacts/kg --persist-neo4j
```

## Local Setup

### Prerequisites

- Python 3.13+
- [Poetry](https://python-poetry.org/docs/#installation)
- Docker and Docker Compose for Neo4j

### Install

```bash
git clone https://github.com/ethantenison/riskfolio-graphrag-agent.git
cd riskfolio-graphrag-agent
poetry install
```

### Configure

Set the Neo4j and source-directory environment variables in `.env` or your shell.

### Start Neo4j

```bash
docker compose up -d
```

### Run the KG pipeline

```bash
poetry run riskfolio-agent kg-run --source-dir /Users/et/Desktop/Data_Projects/Riskfolio-Lib --artifact-dir artifacts/kg
```

### Run the existing app surfaces

```bash
poetry run riskfolio-agent serve --host 127.0.0.1 --port 8000
poetry run riskfolio-agent gradio --host 127.0.0.1 --port 7860
```

## Evaluation

The repository has two evaluation layers:

- answer and retrieval evaluation in [src/riskfolio_graphrag_agent/eval](src/riskfolio_graphrag_agent/eval),
- graph-quality evaluation in [src/riskfolio_graphrag_agent/evaluation/graph_quality.py](src/riskfolio_graphrag_agent/evaluation/graph_quality.py).

## Documentation

- [docs/kg-pipeline-design.md](docs/kg-pipeline-design.md) — KG pipeline design, data model, Cypher model, semantic export, and evaluation
- [docs/architecture_module_map.md](docs/architecture_module_map.md) — architecture boundaries and package ownership
- [docs/retrieval_methods.md](docs/retrieval_methods.md) — retrieval mode descriptions, formulas, performance profiles, and troubleshooting
- [docs/quickstart.md](docs/quickstart.md) — concise local validation commands

## Development

```bash
poetry run pytest -q
poetry run ruff check src tests
poetry run ruff format src tests
```

## Known Limitations

- The pipeline currently uses a structurally honest heuristic open extractor as the default, not a production extraction model.
- Dense retrieval may run with hash-based embeddings in fallback or default demo configurations; not strong evidence of semantic retrieval quality.
- The Neo4j fallback backend for dense retrieval is lexical search over `Chunk` text, not a true vector index.
- Evaluation metrics are mostly heuristic overlap and support proxies, useful for regression tracking but not benchmark-grade evidence.
- Graph retrieval quality depends on how well the promoted graph (CanonicalEntity, Assertion, OntologyClass nodes) has been populated via `kg-run`.

## License

[MIT](LICENSE)
