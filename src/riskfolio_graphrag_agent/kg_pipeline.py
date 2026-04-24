"""Orchestrate the redesigned knowledge graph induction pipeline.

This module coordinates the new architecture from open extraction through
canonicalization, schema induction, retrieval-graph materialization, semantic
export, and graph-quality reporting. It produces explicit intermediate
artifacts so reviewers can inspect extraction truth, canonicalization decisions,
and ontology commitments separately.

Inputs are ingestion-layer `Document` chunks. Outputs are JSON and Turtle
artifacts plus an in-memory `KnowledgeGraphRunResult` for further processing.

This module does not answer user questions or replace runtime retrieval by
itself.
"""

from __future__ import annotations

import json
from pathlib import Path

from riskfolio_graphrag_agent.canonicalization.pipeline import CanonicalizationPipeline
from riskfolio_graphrag_agent.evaluation.graph_quality import evaluate_graph_quality
from riskfolio_graphrag_agent.extraction.pipeline import ChunkOpenExtractorProtocol, HeuristicOpenExtractor
from riskfolio_graphrag_agent.graph_materialization.pipeline import (
    GraphMaterializationPipeline,
    write_materialized_graph,
)
from riskfolio_graphrag_agent.ingestion.loader import Document
from riskfolio_graphrag_agent.kg_models import KnowledgeGraphRunResult
from riskfolio_graphrag_agent.schema_induction.pipeline import SchemaInductionPipeline
from riskfolio_graphrag_agent.semantic_export.pipeline import SemanticExportPipeline


class KnowledgeGraphPipeline:
    """Run the redesigned end-to-end KG induction pipeline."""

    def __init__(self, extractor: ChunkOpenExtractorProtocol | None = None) -> None:
        """Initialize the pipeline with configurable extraction.

        Args:
            extractor: Optional chunk-level extractor. When omitted, the
                heuristic extractor remains the default vertical slice.
        """
        self._extractor = extractor or HeuristicOpenExtractor()
        self._canonicalizer = CanonicalizationPipeline()
        self._schema_inducer = SchemaInductionPipeline()
        self._materializer = GraphMaterializationPipeline()
        self._semantic_exporter = SemanticExportPipeline()

    def run(
        self,
        *,
        documents: list[Document],
        artifact_dir: str | Path,
        persist_neo4j: bool = False,
        neo4j_uri: str | None = None,
        neo4j_user: str | None = None,
        neo4j_password: str | None = None,
        drop_existing: bool = False,
    ) -> KnowledgeGraphRunResult:
        """Run the pipeline and persist reviewable artifacts.

        Args:
            documents: Chunked ingestion documents.
            artifact_dir: Directory where intermediate artifacts are written.
            persist_neo4j: Whether to write the promoted graph into Neo4j.
            neo4j_uri: Neo4j URI used when `persist_neo4j` is true.
            neo4j_user: Neo4j username used when `persist_neo4j` is true.
            neo4j_password: Neo4j password used when `persist_neo4j` is true.
            drop_existing: Whether to clear Neo4j before writing.

        Returns:
            An aggregate in-memory result for the run.
        """
        artifact_root = Path(artifact_dir)
        semantic_dir = artifact_root / "semantic"
        artifact_root.mkdir(parents=True, exist_ok=True)
        semantic_dir.mkdir(parents=True, exist_ok=True)

        extractions = self._extractor.extract_documents(documents)
        canonicalization = self._canonicalizer.run(extractions)
        schema_induction = self._schema_inducer.run(extractions, canonicalization)
        write_plan = self._materializer.run(extractions, canonicalization, schema_induction)
        semantic_export = self._semantic_exporter.run(schema_induction, write_plan)
        graph_quality = evaluate_graph_quality(
            extractions=extractions,
            canonicalization=canonicalization,
            schema_induction=schema_induction,
            write_plan=write_plan,
        )

        artifact_paths = {
            "extractions": self._write_json(
                artifact_root / "extractions.json",
                [item.model_dump(mode="json") for item in extractions],
            ),
            "canonicalization": self._write_json(
                artifact_root / "canonicalization.json",
                canonicalization.model_dump(mode="json"),
            ),
            "schema_candidates": self._write_json(
                artifact_root / "schema_candidates.json",
                schema_induction.model_dump(mode="json"),
            ),
            "schema_review": self._write_text(artifact_root / "schema_review.md", schema_induction.review_markdown),
            "materialized_graph": self._write_json(
                artifact_root / "materialized_graph.json",
                write_plan.model_dump(mode="json"),
            ),
            "graph_quality": self._write_json(
                artifact_root / "graph_quality.json",
                graph_quality.model_dump(mode="json"),
            ),
            "ontology_ttl": self._write_text(semantic_dir / "ontology.ttl", semantic_export.ontology_turtle),
            "instances_ttl": self._write_text(semantic_dir / "instances.ttl", semantic_export.instances_turtle),
        }

        if persist_neo4j:
            if not all([neo4j_uri, neo4j_user, neo4j_password]):
                raise ValueError("Neo4j connection settings are required when persist_neo4j=True.")
            write_materialized_graph(
                neo4j_uri=str(neo4j_uri),
                neo4j_user=str(neo4j_user),
                neo4j_password=str(neo4j_password),
                write_plan=write_plan,
                drop_existing=drop_existing,
            )

        result = KnowledgeGraphRunResult(
            extractions=extractions,
            canonicalization=canonicalization,
            schema_induction=schema_induction,
            write_plan=write_plan,
            semantic_export=semantic_export,
            graph_quality=graph_quality,
            artifact_paths=artifact_paths,
        )
        artifact_paths["run_summary"] = self._write_json(
            artifact_root / "run_summary.json",
            result.model_dump(mode="json"),
        )
        result.artifact_paths = artifact_paths
        return result

    def _write_json(self, path: Path, payload: object) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return str(path)

    def _write_text(self, path: Path, text: str) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return str(path)
