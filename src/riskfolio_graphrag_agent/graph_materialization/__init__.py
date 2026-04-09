"""Neo4j materialization stage for the redesigned KG induction pipeline."""

from riskfolio_graphrag_agent.graph_materialization.pipeline import (
    GraphMaterializationPipeline,
    write_materialized_graph,
)

__all__ = ["GraphMaterializationPipeline", "write_materialized_graph"]
