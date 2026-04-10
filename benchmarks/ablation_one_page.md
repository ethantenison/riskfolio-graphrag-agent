# One-Page Ablation: Retrieval/Router Upgrade Impact

- Updated: 2026-04-10T14:32Z
- Baseline report: eval_results.json
- Clean rebuilt OpenAI report: benchmarks/eval_results_openai_rebuilt.json
- Clean rebuilt hash report: benchmarks/eval_results_hash_rebuilt.json
- Sample set: benchmarks/eval_samples_v1.json (25 samples)

## Executive Summary

- The earlier large OpenAI decline was caused by an invalid comparison: OpenAI query embeddings were evaluated against a non-rebuilt index.
- After rebuilding the vector index cleanly, OpenAI outperforms the baseline and the rebuilt hash index on recall, precision, and faithfulness.
- The remaining weak area is not dense retrieval quality. It is graph-grounding and multi-hop support, which remain low across rebuilt runs.

## Comparison A: Baseline vs Clean Rebuilt OpenAI Index

- Before embedding provider: hash
- After embedding provider: openai

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| context_recall | 0.6109 | 0.7190 | +0.1081 |
| context_precision | 0.2647 | 0.2844 | +0.0197 |
| answer_faithfulness | 0.8400 | 0.8800 | +0.0400 |
| answer_relevance | 0.8673 | 0.8937 | +0.0264 |
| grounding | 0.8601 | 0.7869 | -0.0732 |
| multi_hop_accuracy | 0.1650 | 0.1465 | -0.0185 |
| avg_latency_ms | 280.730 | 584.328 | +303.597 |
| estimated_cost_usd | 0.000200 | 0.000200 | +0.000000 |

## Comparison B: Baseline vs Clean Rebuilt Hash Index

- Before embedding provider: hash
- After embedding provider: hash

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| context_recall | 0.6109 | 0.6459 | +0.0350 |
| context_precision | 0.2647 | 0.2496 | -0.0151 |
| answer_faithfulness | 0.8400 | 0.8267 | -0.0133 |
| answer_relevance | 0.8673 | 0.8905 | +0.0232 |
| grounding | 0.8601 | 0.7926 | -0.0675 |
| multi_hop_accuracy | 0.1650 | 0.1447 | -0.0203 |
| avg_latency_ms | 280.730 | 273.558 | -7.173 |
| estimated_cost_usd | 0.000200 | 0.000200 | +0.000000 |

## Comparison C: Clean Rebuilt Hash vs Clean Rebuilt OpenAI

| Metric | Hash Rebuilt | OpenAI Rebuilt | Delta |
|---|---:|---:|---:|
| context_recall | 0.6459 | 0.7190 | +0.0731 |
| context_precision | 0.2496 | 0.2844 | +0.0348 |
| answer_faithfulness | 0.8267 | 0.8800 | +0.0533 |
| answer_relevance | 0.8905 | 0.8937 | +0.0032 |
| grounding | 0.7926 | 0.7869 | -0.0057 |
| multi_hop_accuracy | 0.1447 | 0.1465 | +0.0018 |
| avg_latency_ms | 273.558 | 584.328 | +310.770 |
| estimated_cost_usd | 0.000200 | 0.000200 | +0.000000 |

## Interpretation

- Dense retrieval was not the core weakness after rebuild. A provider-consistent OpenAI index materially improves retrieval quality versus hash.
- The major earlier failure mode was index/provider mismatch, plus an embedding batching bug that exceeded provider request token limits during ingestion.
- Grounding and multi-hop remain weak after both clean rebuilds, which points to graph evidence quality and claim verification as the next strengthening targets.
- The local Neo4j warnings about missing promoted relationships still indicate graph schema/materialization gaps that can depress graph-heavy questions.

## Notes

- Benchmark outputs now default to benchmarks/eval_results.json to keep benchmark artifacts scoped in benchmarks/.
- The OpenAI embedding client now batches requests to stay under provider token limits during full-corpus indexing.
