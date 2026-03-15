# Quickstart (Hiring Review Path)

## Option A: Full smoke profile

```bash
bash scripts/run_integration_smoke.sh
```

Expected outputs:
- `eval_results.json`
- `benchmarks/retrieval_ablation_results.json`
- `artifacts/observability/sli_report.json`

## Option B: Fast local validation

```bash
poetry install
poetry run pytest -q
poetry run riskfolio-agent eval --samples path/to/eval_samples.json
poetry run python scripts/benchmark_retrieval_ablation.py
poetry run riskfolio-agent er-eval
poetry run python scripts/report_observability.py
```
