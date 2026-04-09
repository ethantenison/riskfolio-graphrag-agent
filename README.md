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

This repository is being redesigned away from a shallow, alias-driven graph builder and toward a staged knowledge graph induction system with:

- open chunk-level extraction,
- mention and assertion persistence,
- explicit canonicalization decisions,
- reviewable schema induction,
- controlled Neo4j materialization for retrieval,
- RDF/OWL/SKOS/PROV export over stabilized graph layers,
- graph-quality evaluation in addition to answer-quality evaluation.

The new pipeline is documented in [docs/kg-redesign.md](docs/kg-redesign.md). A migration note for the retired architecture is in [docs/kg-migration.md](docs/kg-migration.md).

> Disclaimer: this repository is a technical system-design and engineering artifact. It does not provide financial advice.

## Current Technical Stance

The redesigned path treats the following as different things:

- source documents and chunks,
- extracted mentions,
- candidate entities, assertions, and events,
- canonical entities and predicates,
- ontology and concept-scheme commitments,
- final promoted retrieval graph.

That separation matters because it keeps provenance, confidence, and review state visible instead of flattening everything into direct canonical edges too early.

## Main Components

| Area | Current responsibility |
|---|---|
| [src/riskfolio_graphrag_agent/ingestion/loader.py](src/riskfolio_graphrag_agent/ingestion/loader.py) | Chunk source code, docs, examples, and tests into provenance-rich `Document` records |
| [src/riskfolio_graphrag_agent/extraction/pipeline.py](src/riskfolio_graphrag_agent/extraction/pipeline.py) | Open extraction into mentions, candidate entities, candidate assertions, and candidate events |
| [src/riskfolio_graphrag_agent/canonicalization/pipeline.py](src/riskfolio_graphrag_agent/canonicalization/pipeline.py) | Canonicalize open extraction outputs into reviewable entity, predicate, and event-type clusters |
| [src/riskfolio_graphrag_agent/schema_induction/pipeline.py](src/riskfolio_graphrag_agent/schema_induction/pipeline.py) | Aggregate open semantic guesses into schema candidates and induced ontology elements |
| [src/riskfolio_graphrag_agent/graph_materialization/pipeline.py](src/riskfolio_graphrag_agent/graph_materialization/pipeline.py) | Materialize a narrower Neo4j retrieval graph with constraints and representative Cypher queries |
| [src/riskfolio_graphrag_agent/semantic_export/pipeline.py](src/riskfolio_graphrag_agent/semantic_export/pipeline.py) | Export ontology and instance/provenance Turtle views using OWL, SKOS, and PROV-O |
| [src/riskfolio_graphrag_agent/evaluation/graph_quality.py](src/riskfolio_graphrag_agent/evaluation/graph_quality.py) | Score graph-quality metrics such as compression, promotion yield, and schema support |
| [src/riskfolio_graphrag_agent/kg_pipeline.py](src/riskfolio_graphrag_agent/kg_pipeline.py) | Orchestrate the end-to-end redesigned KG pipeline and artifact generation |
| [src/riskfolio_graphrag_agent/retrieval/retriever.py](src/riskfolio_graphrag_agent/retrieval/retriever.py) | Existing retrieval stack, pending deeper migration onto the promoted graph |
| [src/riskfolio_graphrag_agent/agent/workflow.py](src/riskfolio_graphrag_agent/agent/workflow.py) | Existing plan-retrieve-reason-verify orchestration |

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

## Commands

### Recommended redesigned path

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

### Legacy command

```bash
poetry run riskfolio-agent build-graph
```

`build-graph` is retained temporarily as a compatibility path for the older deterministic builder. It is not the recommended architecture and should not be treated as the serious KG induction path.

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

### Run the redesigned KG pipeline

```bash
poetry run riskfolio-agent kg-run --source-dir /Users/et/Desktop/Data_Projects/Riskfolio-Lib --artifact-dir artifacts/kg
```

### Run the existing app surfaces

```bash
poetry run riskfolio-agent serve --host 127.0.0.1 --port 8000
poetry run riskfolio-agent gradio --host 127.0.0.1 --port 7860
```

## Evaluation

The repository currently has two evaluation layers:

- answer and retrieval evaluation in [src/riskfolio_graphrag_agent/eval](src/riskfolio_graphrag_agent/eval),
- graph-quality evaluation in [src/riskfolio_graphrag_agent/evaluation/graph_quality.py](src/riskfolio_graphrag_agent/evaluation/graph_quality.py).

The graph-quality report produced by `kg-run` currently measures pipeline structure and promotion behavior. It is the first step toward fuller extraction, canonicalization, and retrieval-lift evaluation described in [docs/kg-redesign.md](docs/kg-redesign.md).

## Documentation

- [docs/kg-redesign.md](docs/kg-redesign.md) — redesign goals, data model, pipeline, Cypher model, semantic export, and evaluation plan
- [docs/kg-migration.md](docs/kg-migration.md) — what is being retired and which compatibility is intentionally dropped
- [docs/architecture_module_map.md](docs/architecture_module_map.md) — architecture boundaries and package ownership
- [docs/quickstart.md](docs/quickstart.md) — concise local validation commands

## Development

```bash
poetry run pytest -q
poetry run ruff check src tests
poetry run ruff format src tests
```

## Known Limitations

- The redesigned pipeline currently includes a structurally honest heuristic open extractor as the default vertical slice, not a production extraction model.
- Retrieval and UI surfaces still consume parts of the legacy graph path and need further migration onto the promoted graph.
- Semantic export is cleaner than the old direct registry dump, but it does not yet enforce SHACL validation or claim full ontology rigor.

## License

[MIT](LICENSE)
