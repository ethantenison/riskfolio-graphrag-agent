# Benchmark Artifacts Policy

This directory stores small, stable benchmark inputs and outputs that are safe to version.

## What should be committed

- Canonical benchmark inputs and baselines (for example `eval_samples_v1.json`).
- Stable comparison outputs (for example ablation `eval_*_top8.json` and `ablation_summary_top8.json`).
- Human-readable benchmark summaries (`*.md`) and query examples (`*.rq`).

## What should not be committed

- Local vector-store database artifacts and runtime caches.
- Large generated stores such as Chroma SQLite files.

Ignored examples are managed in the root `.gitignore`, including:

- `benchmarks/.chroma_hash_clean/`
- `benchmarks/.chroma_openai_clean/`
- `benchmarks/**/*.sqlite3`
- `benchmarks/**/*.sqlite3-*`

## Optional noise-reduction policy

If you want fewer top-level benchmark files, keep one canonical latest file:

- `eval_results_semantic_expansion_latest.json`

and archive other run variants under `artifacts/comparisons/<date>/`.

Note: `artifacts/comparisons` is currently ignored by Git in this repository, so archived files are local-only unless ignore rules are changed.
