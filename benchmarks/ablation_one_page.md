# One-Page Ablation: Retrieval/Router Upgrade Impact

- Generated: 2026-04-10T14:20:51.128296+00:00
- Baseline report: eval_results.json
- New default report: benchmarks/eval_results.json
- Controlled hash report: benchmarks/eval_results_hash_control.json
- Sample set: benchmarks/eval_samples_v1.json (25 samples)

## Executive Summary

- A fair implementation comparison is the controlled hash run (hash vs hash).
- The openai-default run is included for operational visibility but is not apples-to-apples with the old hash baseline.
- Controlled hash deltas show higher recall/relevance and lower latency, with drops in precision/grounding/multi-hop that warrant follow-up tuning.

## Comparison A: Baseline vs New Default (Provider Changed)

- Before embedding provider: hash
- After embedding provider: openai

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| context_recall | 0.6109 | 0.5525 | -0.0584 |
| context_precision | 0.2647 | 0.1711 | -0.0936 |
| answer_faithfulness | 0.8400 | 0.7800 | -0.0600 |
| answer_relevance | 0.8673 | 0.8862 | +0.0189 |
| grounding | 0.8601 | 0.7817 | -0.0784 |
| multi_hop_accuracy | 0.1650 | 0.1071 | -0.0579 |
| avg_latency_ms | 280.730 | 958.337 | +677.607 |
| estimated_cost_usd | 0.000200 | 0.000200 | +0.000000 |

## Comparison B: Controlled (Hash vs Hash)

- Before embedding provider: hash
- After embedding provider: hash

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| context_recall | 0.6109 | 0.6259 | +0.0150 |
| context_precision | 0.2647 | 0.2456 | -0.0191 |
| answer_faithfulness | 0.8400 | 0.8267 | -0.0133 |
| answer_relevance | 0.8673 | 0.8900 | +0.0227 |
| grounding | 0.8601 | 0.7913 | -0.0688 |
| multi_hop_accuracy | 0.1650 | 0.1447 | -0.0203 |
| avg_latency_ms | 280.730 | 270.739 | -9.991 |
| estimated_cost_usd | 0.000200 | 0.000200 | +0.000000 |

## Domain Deltas (Controlled Hash)

| Domain | Recall Before | Recall After | Recall Delta | Precision Before | Precision After | Precision Delta |
|---|---:|---:|---:|---:|---:|---:|
| estimation | 0.5750 | 0.6000 | +0.0250 | 0.2871 | 0.2163 | -0.0708 |
| optimization | 0.6217 | 0.6217 | +0.0000 | 0.2626 | 0.2263 | -0.0363 |
| portfolio-construction | 0.7500 | 0.7500 | +0.0000 | 0.3093 | 0.2862 | -0.0231 |
| reporting | 0.5417 | 0.5833 | +0.0417 | 0.2288 | 0.2596 | +0.0307 |
| risk-measures | 0.6333 | 0.6333 | +0.0000 | 0.2610 | 0.2572 | -0.0038 |

## Notes

- Benchmark outputs now default to benchmarks/eval_results.json to keep benchmark artifacts scoped in benchmarks/.
- Neo4j warnings about missing promoted relationships indicate graph schema variance in this local DB and can affect graph-related metrics.
