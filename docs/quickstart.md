# Quickstart

## Option A: Redesigned KG path

```bash
poetry install
poetry run pytest -q
poetry run riskfolio-agent kg-run --source-dir /path/to/Riskfolio-Lib --artifact-dir artifacts/kg
```

Example:
```bash
poetry run riskfolio-agent kg-run \
  --source-dir /Users/et/Desktop/Data_Projects/Riskfolio-Lib \
  --artifact-dir artifacts/kg \
  --persist-neo4j \
  --drop-existing
```

Expected outputs:

- `artifacts/kg/extractions.json`
- `artifacts/kg/canonicalization.json`
- `artifacts/kg/schema_candidates.json`
- `artifacts/kg/schema_review.md`
- `artifacts/kg/materialized_graph.json`
- `artifacts/kg/graph_quality.json`
- `artifacts/kg/semantic/ontology.ttl`
- `artifacts/kg/semantic/instances.ttl`

## Option B: Existing app and answer-eval path

```bash
poetry install
poetry run pytest -q
poetry run riskfolio-agent eval --samples benchmarks/eval_samples_v1.json
poetry run riskfolio-agent er-eval
poetry run python scripts/report_observability.py
```

## Legacy compatibility path

```bash
poetry run riskfolio-agent build-graph
```

Use this only if you specifically need the older deterministic graph builder for comparison or migration work.

