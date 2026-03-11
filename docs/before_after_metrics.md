# RAG System Metrics

Measured against the live Neo4j graph (3,590 nodes, 9,478 edges) using the deterministic evaluation harness. Retrieval mode: **sparse** (ablation winner). ER audit detail: `artifacts/er/er_audit.json`.

## Retrieval Quality

| Metric | Value |
|---|---:|
| Context recall | 0.55 |
| Context precision | 0.1564 |
| Answer faithfulness | 1.0000 |
| Answer relevance | 0.8525 |
| Grounding | 0.7734 |
| Multi-hop accuracy | 1.0000 |

## Knowledge Graph Integrity

| Metric | Value |
|---|---:|
| SHACL pass rate | 1.0000 |
| ER precision | 1.0000 |
| ER recall | 1.0000 |
| ER F1 | 1.0000 |

## Link Prediction

| Metric | Value |
|---|---:|
| MRR | 1.0000 |
| Hits@3 | 1.0000 |
| Hits@10 | 1.0000 |

## Operational

| Metric | Value |
|---|---:|
| Avg latency (ms) | 42.6 |
| Estimated cost / query (USD) | 0.00020 |

## Retrieval Mode Ablation

| Mode | Recall | Precision |
|---|---:|---:|
| sparse | 0.25 | 0.0757 |
| graph | 0.23 | 0.0775 |
| dense | 0.24 | 0.0750 |
| hybrid_rerank | 0.12 | 0.0518 |

Sparse is the default. The router automatically promotes code/API queries to sparse and graph-traversal queries to graph mode. Full results: `benchmarks/retrieval_ablation_results.md`.
