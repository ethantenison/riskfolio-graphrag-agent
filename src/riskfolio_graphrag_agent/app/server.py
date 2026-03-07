"""FastAPI application server for the GraphRAG agent.

This module wires the retrieval and agent layers into HTTP endpoints so the
demo can be served as a simple REST API.  A Gradio or Streamlit front-end can
be added later by importing :func:`create_app` and mounting it.

Endpoints (planned)
-------------------
GET  /health           – liveness probe.
POST /query            – ask a question, receive a cited answer.
GET  /graph/stats      – node and edge counts from Neo4j.

This module currently provides **stub** implementations.  Replace the
``# TODO`` sections with real FastAPI route handlers.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def create_app() -> object:
    """Create and return a configured FastAPI application instance.

    Returns:
        A ``FastAPI`` application object (stub returns a plain object).
    """
    # TODO: from fastapi import FastAPI; app = FastAPI(...)
    # TODO: register /health, /query, /graph/stats routes
    logger.info("create_app called (stub – FastAPI not yet wired).")
    return _StubApp()


class _StubApp:
    """Placeholder until FastAPI is added as a dependency."""

    def __repr__(self) -> str:
        return "StubApp(FastAPI placeholder)"
