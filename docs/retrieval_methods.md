# Retrieval Methods

This document describes the four retrieval modes and how they work as of the current implementation.

## Overview

The retrieval layer in `src/riskfolio_graphrag_agent/retrieval/retriever.py` orchestrates evidence collection for user queries by supporting four distinct retrieval modes:

1. **Dense** — embedding-based semantic similarity
2. **Sparse** — lexical token matching and keyword search
3. **Graph** — entity-seeded traversal with assertion-aware expansion
4. **Hybrid Rerank** — merged dense and sparse results with graph-contextualized boosting

Each mode operates in two stages:
1. Collect initial hits from the selected strategy
2. Expand each hit with local graph context and apply optional reranking boosts

## Dense Retrieval

**Strategy**: Embedding-based semantic similarity search.

**How it works**:
- Query expansion: Generates multiple natural-language variants of the original query
- Candidate pool: Fetches `2×top_k` results from the vector store for each variant
- Merging: Combines results using reciprocal-rank fusion to handle score misalignment across vector backends
- Graph expansion: Each result is enriched with related entities and graph neighbours
- Reranking: Evidence boost applied as `0.07×entity_signal + 0.03×neighbour_signal + 0.08×coverage_signal`

**Use case**: Broad semantic similarity; best for conceptual questions like "What is Hierarchical Risk Parity?"

**Strengths**:
- Fast and scalable
- Works well for out-of-vocabulary or paraphrased queries
- Embedding model captures semantic relationships

**Weaknesses**:
- May miss exact terminology or rare domain terms
- Embedding quality is bounded by the model used

---

## Sparse Retrieval

**Strategy**: Direct lexical token matching against Neo4j chunk nodes.

**How it works**:
- Token extraction: Decomposes query into multi-character tokens (2+ chars)
- Chunk matching: Searches Neo4j `Chunk` nodes for token presence
- Scoring: Orders results by token frequency / BM25-like scoring
- No reranking: Returns results as-is; no graph expansion or evidence boosts

**Use case**: Term-driven queries with specific acronyms or exact phrases; e.g., "CVaR" or "Ledoit-Wolf shrinkage"

**Strengths**:
- Precise for exact terminology
- Deterministic and easily explainable
- No embedding model required

**Weaknesses**:
- Fails on synonyms or semantic paraphrasing
- Limited to lexical overlap

---

## Graph Retrieval

**Strategy**: Knowledge graph seeding and multi-hop assertion-aware expansion.

**How it works**:

### Stage 1: Candidate Collection (candidate_k = 3×top_k)

Graph retrieval builds a broader candidate pool (3× the final `top_k`) by combining four retrieval pathways:

#### a) **Promoted Graph Seed Hits**
- Matches `CanonicalEntity` nodes whose `preferred_label` or `normalized_label` contain query tokens
- Collects chunks supported by `Assertion` nodes linked to matched entities
- Score: Max confidence of supporting assertions

#### b) **Promoted Graph Hop Expansion** (Ontology-aware)
- Finds `OntologyClass` nodes matching query tokens or class definitions
- Traverses `INSTANCE_OF` edges to find `CanonicalEntity` instances
- Collects chunks supported by `Assertion` nodes
- Score: Max assertion confidence

#### c) **Promoted Graph Bridge Hits** (Assertion-bridged peer entities)
- Seeds from matching `CanonicalEntity` nodes
- Traverses **two assertion hops**: seed entity → assertion → peer entity → assertion → chunk
- Bridges confidence-weighted: `0.35×a1.confidence + 0.65×a2.confidence`
- Extends coverage by finding peer entities mentioned in same assertions

#### d) **Sparse Backfill** (Keyword matching)
- Adds lexical matches to fill coverage gaps
- Candidate pool: `max(top_k, candidate_k // 2)` results
- Ensures rare terminology and domain-specific terms aren't lost

### Stage 2: Graph Expansion and Reranking
- Each hit is enriched with related entities and graph neighbours
- Reranking boost: `0.10×entity_signal + 0.05×neighbour_signal + 0.15×coverage_signal`

**Use case**: Complex multi-hop questions requiring domain relationships; e.g., "What are the relationships between Hierarchical Risk Parity and alternative risk measures?"

**Strengths**:
- Captures semantic relationships via the knowledge graph
- Bridge expansion reaches peer entities and cross-linked concepts
- Assertion confidence allows fine-grained ranking
- Sparse backfill ensures no terminology is completely missed

**Weaknesses**:
- Requires well-populated Neo4j graph with `Assertion` and `OntologyClass` nodes
- Slower than dense or sparse (multiple Cypher queries)
- Graph quality directly impacts retrieval quality

---

## Hybrid Rerank

**Strategy**: Merge dense and sparse results from a broader candidate pool, then apply graph-contextualized boosting.

**How it works**:

### Stage 1: Candidate Collection
- Dense search: `2×top_k` results from embedding similarity
- Sparse search: `2×top_k` results from token matching
- Merging: Reciprocal-rank fusion combines both result sets
- Candidate pool: Up to `4×top_k` unique results before final truncation

### Stage 2: Graph Expansion and Reranking
- Each merged hit is enriched with related entities and graph neighbours
- Three signals contribute to the final score:
  - **Entity signal**: Logarithmic scaling of related entity count; max boost ~0.06
  - **Neighbour signal**: Logarithmic scaling of graph neighbours; max boost ~0.04
  - **Coverage signal**: Fraction of query tokens appearing in text; max boost ~0.05
- Evidence boost: `0.11×entity_signal + 0.07×neighbour_signal + 0.09×coverage_signal`
- Final score: `0.85×merged_score + evidence_boost`

**Use case**: Balanced precision and recall for general-purpose queries; default mode.

**Strengths**:
- Combines complementary strengths of dense and sparse retrieval
- Broad candidate pool reduces missed results
- Graph context provides interpretable ranking signals
- Fast and reliable for a wide range of query types

**Weaknesses**:
- Moderate computational cost (multiple searches)
- Performance depends on both embedding model and keyword matching quality

---

## Evidence Boost Signals

All retrieval modes except sparse apply three optional reranking signals:

### Entity Signal
Measures the richness of entity context around a result.

$$\text{entity\_signal} = \min\left(1.0, \frac{\log(1 + \text{entity\_count})}{\log(6.0)}\right)$$

Saturates at ~6 related entities for the maximum boost.

### Neighbour Signal
Measures graph connectivity around a result.

$$\text{neighbour\_signal} = \min\left(1.0, \frac{\log(1 + \text{neighbour\_count})}{\log(8.0)}\right)$$

Saturates at ~8 graph neighbours for the maximum boost.

### Coverage Signal
Measures how many query tokens appear in the retrieved text.

$$\text{coverage\_signal} = \frac{\text{tokens\_found}}{\text{query\_tokens}}$$

Ranges from 0 (no query tokens found) to 1.0 (all query tokens present).

---

## Scoring and Ranking

### Reciprocal Rank Fusion (Dense and Hybrid)
When merging results from multiple sources, reciprocal rank fusion weights higher-ranked items more heavily while handling score-scale mismatches:

$$\text{final\_score} = \sum \frac{1}{60 + \text{rank}}$$

This ensures deterministic, reproducible scoring regardless of the underlying vector backend (Chroma vs Neo4j).

### Mode-Specific Reranking Weights

| Mode | Base Score Weight | Entity Boost | Neighbour Boost | Coverage Boost | Total Boost |
|------|-------------------|--------------|-----------------|----------------|-------------|
| Dense | 0.88 | 0.07 | 0.03 | 0.08 | 0.18 |
| Graph | 0.70 | 0.10 | 0.05 | 0.15 | 0.30 |
| Hybrid Rerank | 0.85 | 0.11 | 0.07 | 0.09 | 0.27 |

Graph mode gives the highest weight to evidence boosts (30% of final score), reflecting greater reliance on graph structure. Dense mode is conservative (18% boost), trusting the embedding score. Hybrid rerank balances both (27% boost).

---

## Query Tokenization

Different retrieval pathways use slightly different tokenization to match their strategy:

### Dense Query Tokenization
- Standard `re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}")` (2+ character tokens)
- No synonym expansion; used as-is for embedding queries
- Capped at 12 tokens

### Lexical/Graph Query Tokenization
- Same base tokenization as dense
- **Extended with synonyms** for common domain terms:
  - `cvar` → `(conditional, tail, risk)`
  - `cdar` → `(drawdown, risk)`
  - `ledoit` → `(shrinkage)`
  - And others as defined in `_LEXICAL_TOKEN_SYNONYMS`
- Prevents query narrowing when exact terminology isn't user-provided

---

## Promoted Graph vs Legacy Fallback

The current implementation **prefers the promoted assertion-aware graph** and falls back to legacy patterns when promoted nodes are unavailable.

### Promoted Pattern (Preferred)
- `CanonicalEntity` with `preferred_label` and `normalized_label`
- `Assertion` nodes with `confidence` scores linking entities to chunks
- `OntologyClass` nodes with `INSTANCE_OF` relationships
- Supports **bridge expansion** for multi-hop reasoning

### Legacy Pattern (Fallback)
- Generic entity nodes with `MENTION` edges
- Direct `Chunk` node matching on raw `name` fields
- Simple ontology-like edges (`IS_SUBTYPE_OF`, `ALTERNATIVE_TO`, `REQUIRES`, etc.)
- Used when promoted nodes are not available

Selection logic:
```python
seed_hits = _promoted_graph_seed_hits(...)
if not seed_hits:
    seed_hits = _graph_seed_hits(...)  # Fallback to legacy

hop_hits = _promoted_graph_hop_expansion(...)
if not hop_hits:
    hop_hits = _graph_hop_expansion(...)  # Fallback to legacy
```

---

## Configuration and Parameters

### HybridRetriever Constructor Arguments

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `top_k` | `int` | 5 | Final number of results to return |
| `vector_store_backend` | `str` | `"neo4j"` | Vector backend: `"neo4j"` or `"chroma"` |
| `chroma_persist_dir` | `str` | `".chroma"` | Persistence directory for Chroma |
| `retrieval_mode` | `RetrievalMode` | `"hybrid_rerank"` | Default mode: `"dense"`, `"sparse"`, `"graph"`, or `"hybrid_rerank"` |
| `embedding_provider` | `EmbeddingProvider` | `HashEmbeddingProvider` | Embedding model for dense search |

### Candidate Pool Sizing

| Mode | Formula | Example (top_k=5) |
|------|---------|-------------------|
| Dense | `2×top_k` | 10 candidates |
| Sparse | `top_k` | 5 candidates |
| Graph | `3×top_k` | 15 candidates |
| Hybrid | `2×top_k` per source | 10 dense + 10 sparse = 20 candidates |

---

## Non-Goals

The retrieval layer does **not**:
- Chunk source files (handled by ingestion layer)
- Build the Neo4j graph schema (handled by graph materialization)
- Generate final natural-language answers (handled by agent layer)
- Automatically choose retrieval mode (caller specifies or uses default)
- Run cross-encoders or learned rerankers
- Translate natural language to Cypher queries

---

## Example Usage

```python
from riskfolio_graphrag_agent.retrieval.retriever import HybridRetriever

# Initialize retriever
retriever = HybridRetriever(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password",
    top_k=8,
    vector_store_backend="neo4j",
    retrieval_mode="hybrid_rerank"  # or "dense", "sparse", "graph"
)

try:
    # Default mode (hybrid_rerank)
    results = retriever.retrieve("What is Hierarchical Risk Parity?")
    for item in results:
        print(f"{item.score:.4f} | {item.source_path} | {item.content[:80]}")

    # Override mode for this query
    graph_results = retriever.retrieve(
        "How does HRP relate to other risk measures?",
        mode_override="graph"
    )

finally:
    retriever.close()
```

---

## Performance Considerations

### Latency Profile

From recent evaluation runs:

- **Dense**: ~200-400ms (single vector search + graph expansion)
- **Sparse**: ~150-300ms (Cypher token lookup + graph expansion)
- **Graph**: ~800-1500ms (four Cypher queries + merging + graph expansion)
- **Hybrid Rerank**: ~600-1000ms (dense + sparse in parallel + graph expansion)

Graph mode is slower due to multiple Cypher queries but provides richer context. For latency-sensitive applications, dense or hybrid rerank is preferred.

### Memory Usage

- Dense: Requires full embedding model in memory (~500MB for OpenAI embeddings)
- Sparse: Graph traversal memory (varies by Neo4j scale; typically <100MB)
- Graph: Multiple intermediate result sets (candidate_k * 4 in worst case)
- Hybrid: Combined dense + sparse footprint

---
