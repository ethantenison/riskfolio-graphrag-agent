# Before/After Metrics

Baseline values are from the original `eval_results.json` profile. After values reflect the expanded scorecard in the upgraded pipeline.
ER values in this table come from the eval scorecard sample. The standalone ER audit sample is available in `artifacts/er/er_audit.json`.

| Metric | Before | After |
|---|---:|---:|
| Context recall | 0.55 | 0.55 |
| Context precision | 0.92 | 0.92 |
| Answer faithfulness | 0.4051 | 0.4051 |
| Answer relevance | 1.00 | 1.00 |
| Grounding | n/a | 0.4210 |
| Multi-hop accuracy | n/a | 0.5200 |
| ER precision | n/a | 1.0000 |
| ER recall | n/a | 1.0000 |
| ER F1 | n/a | 1.0000 |
| Link prediction MRR | n/a | 0.7100 |
| Link prediction Hits@3 | n/a | 1.0000 |
| Link prediction Hits@10 | n/a | 1.0000 |
| Avg latency (ms) | n/a | 112.3 |
| Estimated cost (USD) | n/a | 0.00093 |

## Retrieval Ablation Snapshot

See `benchmarks/retrieval_ablation_results.md` for mode-by-mode results:
- dense
- sparse
- graph
- hybrid_rerank (current runtime default)
