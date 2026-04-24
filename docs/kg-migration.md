# KG Migration and Deprecation

## Retired or Demoted Components

The following legacy mechanisms are no longer the target architecture:

- hardcoded `DOMAIN_ALIASES` as the primary semantic mechanism,
- predefined node-label and edge-type registries as the main extraction boundary,
- taxonomy edge emission as a substitute for induced schema,
- direct chunk-to-canonical-edge graph writes,
- mixed ontology and instance export in a single decorative RDF dump.

These components still exist temporarily for compatibility and comparison work, but they are not the recommended path.

## New Command Surface

Preferred command:

```bash
poetry run riskfolio-agent kg-run --artifact-dir artifacts/kg --source-dir /path/to/Riskfolio-Lib
```

Legacy command retained temporarily:

```bash
poetry run riskfolio-agent build-graph
```

## Artifact Changes

Old graph path emphasized direct Neo4j graph state.

New graph path writes explicit staged artifacts:

- `extractions.json`
- `canonicalization.json`
- `schema_candidates.json`
- `schema_review.md`
- `materialized_graph.json`
- `graph_quality.json`
- `semantic/ontology.ttl`
- `semantic/instances.ttl`

## Compatibility Intentionally Dropped

- The redesigned extraction stage does not constrain itself to the old pre-whitelisted ontology.
- The redesigned schema path does not rely on alias expansion to look sophisticated.
- The new semantic export separates ontology commitments from instance provenance.

## Short-Term Compatibility That Still Exists

- `build-graph` remains available as a legacy comparison path.
- Existing retrieval, agent, and app surfaces still rely partly on the older graph path.
- Existing answer-quality evaluation remains in place while graph-quality metrics grow.

## Migration Priority

1. Use `kg-run` to generate reviewable graph artifacts.
2. Promote retrieval queries onto the materialized graph rather than the legacy alias graph.
3. Move UI graph visualization and graph stats onto the redesigned graph surfaces.
4. Retire the legacy builder after downstream consumers no longer depend on it.