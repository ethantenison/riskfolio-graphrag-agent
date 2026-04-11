"""Open extraction stage for the redesigned KG induction pipeline.

This package owns chunk-level open-world extraction into mention, assertion,
and event records. It preserves provenance aggressively and does not commit to
final ontology or retrieval schema decisions.
"""

from riskfolio_graphrag_agent.extraction.pipeline import HeuristicOpenExtractor

__all__ = ["HeuristicOpenExtractor"]
