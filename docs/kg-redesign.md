# KG Redesign

## Goals

The redesign replaces the old deterministic, alias-driven graph builder with a staged knowledge graph induction system that is credible as knowledge engineering work rather than as a demo taxonomy dump.

The primary goals are:

- perform open extraction before schema mapping,
- preserve mention-level provenance and assertion evidence,
- separate extraction truth from canonicalization and ontology commitments,
- materialize a practical retrieval graph for Neo4j,
- export a semantically disciplined OWL/SKOS/PROV view,
- measure graph quality, not just answer quality.

## Non-Goals

- Preserve the old alias dictionary and taxonomy edges as the core mechanism.
- Pretend that expanded regexes constitute ontology induction.
- Collapse extraction, canonicalization, schema induction, and graph materialization into one builder step.
- Over-claim formal ontology rigor while the ontology is still induced and evolving.

## Graph Layers

### 1. Source layer

- `SourceDocumentRecord`
- `ChunkRecord`

Purpose:

- retain original source boundaries,
- preserve chunk hashes and source metadata,
- provide stable references for provenance and review.

### 2. Extraction layer

- `MentionRecord`
- `CandidateEntityRecord`
- `CandidateAssertionRecord`
- `CandidateEventRecord`
- `EvidenceSpan`

Purpose:

- capture what the extractor believes was said in each chunk,
- preserve evidence spans and confidence,
- avoid early normalization into a fixed ontology.

### 3. Canonical semantic layer

- `CanonicalEntity`
- `CanonicalPredicate`
- `CanonicalEventType`
- `CanonicalizationDecision`

Purpose:

- cluster open extraction outputs,
- make entity and predicate decisions explicit and reviewable,
- preserve alternatives and rationales.

### 4. Ontology and schema layer

- `SchemaInductionCandidate`
- `OntologyClass`
- `OntologyProperty`
- `ConceptScheme`

Purpose:

- aggregate semantic guesses into reviewable schema proposals,
- distinguish schema proposals from stabilized commitments,
- support versioned ontology decisions.

### 5. Retrieval materialization layer

- `MaterializedNode`
- `MaterializedEdge`
- `GraphWritePlan`

Purpose:

- expose a practical property graph to Neo4j,
- keep evidence-bearing assertions reified,
- avoid upstream noise from becoming direct retrieval edges.

## Extraction Flow

1. Ingestion creates chunked `Document` records.
2. Open extraction emits mention, entity, assertion, and event candidates with free-text semantic guesses.
3. Evidence spans and extraction metadata are persisted with each record.
4. No hardcoded domain ontology is required at this stage.

The current vertical slice ships with a heuristic extractor for honesty and testability. The interface is designed so higher-quality LLM-backed extraction can replace it without changing downstream contracts.

## Canonicalization Flow

Canonicalization operates after extraction and before schema stabilization.

The pipeline is structured to combine:

- lexical similarity,
- embedding similarity hooks,
- provenance and local graph context,
- optional LLM adjudication,
- human review.

The current implementation provides the structural scaffolding and a lexical clustering baseline. Every clustering decision is explicit in `CanonicalizationDecision` rather than hidden in alias maps.

## Schema Induction Flow

Schema induction aggregates:

- entity `type_guess` values,
- relation `relation_guess` values,
- event `event_type_guess` values.

It produces:

- JSON artifacts for machine review,
- Markdown summaries for human review,
- ontology class and property proposals,
- concept schemes for emergent taxonomies.

This stage is intentionally reviewable and versionable. It is not a silent registry expansion step.

## Final Neo4j Graph Model

The final property graph uses stable IDs and a narrow operational label set:

- `SourceDocument`
- `Chunk`
- `CanonicalEntity`
- `Assertion`
- `OntologyClass`
- `OntologyProperty`
- `ConceptScheme`

Representative relationship types:

- `HAS_CHUNK`
- `INSTANCE_OF`
- `HAS_CONCEPT`
- `ASSERTS_SUBJECT`
- `ASSERTS_OBJECT`
- `ASSERTS_PREDICATE`
- `SUPPORTED_BY`

Key properties:

- `node_id` as the primary stable identifier,
- labels and normalized forms for canonical entities,
- confidence and status on canonical and assertion nodes,
- `source_chunk_id` and `evidence_ids` on assertions,
- `definition` and review status on ontology elements.

Constraint strategy:

- uniqueness constraints on `node_id` per operational label,
- secondary indexes on canonical entity labels and assertion relation guesses.

## RDF, OWL, SKOS, and PROV Mapping Strategy

The redesign separates semantic export into two views:

### Ontology view

- OWL classes for stabilized ontology classes,
- OWL object properties for stabilized ontology properties,
- SKOS concept schemes for emergent class and property families.

### Instance and provenance view

- materialized nodes typed into the project namespace,
- assertions additionally typed as `prov:Entity`,
- `SUPPORTED_BY` edges mapped to `prov:wasDerivedFrom`.

This is a cleaner split than the old direct registry dump, but it still avoids overstating formal semantics while the schema remains induced and evolving.

## Evaluation Plan

Graph-quality evaluation should eventually cover:

- entity extraction quality,
- relation extraction quality,
- canonicalization quality,
- schema stability across runs,
- ontology coherence and validation,
- retrieval lift versus non-graph baselines.

The implemented vertical slice now reports:

- counts of chunks, mentions, candidate entities, assertions, and events,
- canonicalization compression ratio,
- promoted assertion ratio,
- mean assertion confidence,
- mean canonical entity confidence,
- schema support ratio.

## Tradeoffs and Risks

- The current extractor is structurally honest but not yet strong enough to be the final extraction system.
- Retrieval remains partly coupled to the legacy graph until downstream migration lands.
- Schema induction is reviewable, but formal validation and schema version governance still need deeper implementation.
- The semantic export is disciplined enough to be meaningful, but not yet a full ontology management stack.

## What Is Implemented Now

- shared typed models for all graph layers,
- open extraction vertical slice,
- canonicalization scaffolding and decision records,
- schema induction artifacts,
- Neo4j write plans and write functions,
- RDF/OWL/SKOS/PROV export split,
- graph-quality evaluation,
- `kg-run` CLI entry point.

## What Still Needs Work

- stronger extraction models,
- richer canonicalization features such as embeddings and adjudication,
- retrieval migration onto the promoted graph,
- semantic validation beyond lightweight structural export,
- graph-quality regression gating in CI.