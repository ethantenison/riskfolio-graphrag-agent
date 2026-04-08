# Known limitations

This repository is a GraphRAG / hybrid retrieval demo and evaluation scaffold, not a production-ready retrieval platform.

## Current limitations

- Dense retrieval may run with hash-based embeddings in fallback or default demo configurations. That is useful for deterministic testing, but it is not strong evidence of semantic retrieval quality.
- The Neo4j fallback backend is lexical retrieval over `Chunk` text, not a true vector index.
- Hybrid ranking currently uses fixed-score mixing and lightweight graph boosts, not calibrated score normalization or learned reranking.
- Query routing is heuristic and should be treated as a prototype router rather than a robust intent classifier.
- Evaluation metrics are mostly heuristic overlap and support proxies. They are useful for regression tracking, but they are not benchmark-grade evidence on their own.
- The retrieval ablation currently shows that sparse retrieval wins on the published benchmark. Graph retrieval helps some query classes conceptually, but that value is not yet demonstrated convincingly at larger scale.
- The benchmark is versioned and expanded in `benchmarks/eval_samples_v1.json`, but it is still a manually authored benchmark and should be interpreted accordingly.

## How to read the results

- Treat the evaluation harness as a regression and comparison tool first.
- Treat graph-oriented retrieval as an exploratory architecture whose value depends on query class.
- Prefer real embedding backends for serious runs.
- Do not interpret the current metrics as proof of production readiness.
