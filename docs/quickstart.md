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

## Read order for hiring managers

1. `docs/hiring_manager_one_page.md`
2. `docs/capability_matrix.md`
3. `docs/architecture_module_map.md`
4. `docs/before_after_metrics.md`
5. `docs/demo_script_5_7_min.md`
