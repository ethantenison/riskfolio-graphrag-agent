"""Gradio chat interface with Neo4j-backed graph visualisation."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import gradio as gr

from riskfolio_graphrag_agent.agent.workflow import AgentWorkflow
from riskfolio_graphrag_agent.config.settings import Settings
from riskfolio_graphrag_agent.graph.builder import GraphBuilder
from riskfolio_graphrag_agent.graph.semantic_interop import describe_relationship_semantics
from riskfolio_graphrag_agent.retrieval.embeddings import resolve_embedding_provider
from riskfolio_graphrag_agent.retrieval.retriever import HybridRetriever
from riskfolio_graphrag_agent.retrieval.router import QueryToolRouter

_CONTRASTIVE_ARTIFACT_PATH = Path("artifacts/eval/contrastive.json")
_ABLATION_ARTIFACT_PATH = Path("benchmarks/retrieval_ablation_results.json")


def run_query_with_graph(
    question: str,
    top_k: int = 10,
    graph_max_nodes: int = 40,
    graph_max_edges: int = 80,
) -> tuple[str, list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    normalized_question = question.strip()
    if not normalized_question:
        return "Please enter a question.", [], {"nodes": [], "edges": []}, {}

    settings = Settings()
    provider_resolution = resolve_embedding_provider(
        provider_name=settings.embedding_provider,
        embedding_dim=settings.embedding_dim,
        openai_api_key=settings.openai_api_key,
        openai_embedding_model=settings.embedding_model,
        openai_base_url=settings.openai_base_url,
        openai_timeout_seconds=settings.openai_timeout_seconds,
    )

    retriever = HybridRetriever(
        neo4j_uri=settings.neo4j_uri,
        neo4j_user=settings.neo4j_user,
        neo4j_password=settings.neo4j_password,
        top_k=max(1, int(top_k)),
        vector_store_backend=settings.vector_store_backend,
        chroma_persist_dir=settings.chroma_persist_dir,
        embedding_provider=provider_resolution.provider,
        retrieval_mode=settings.retrieval_mode,
    )

    query_router = None
    if settings.adaptive_tool_routing_enabled:
        query_router = QueryToolRouter(
            min_confidence=settings.adaptive_tool_routing_min_confidence,
        )

    llm_generate = None
    if settings.openai_enable_generation and settings.openai_api_key.strip():
        from riskfolio_graphrag_agent.app.server import _make_openai_llm_generate

        llm_generate = _make_openai_llm_generate(settings)

    workflow = AgentWorkflow(
        retriever=retriever,
        model_name=settings.openai_model,
        llm_generate=llm_generate,
        query_router=query_router,
    )

    try:
        state = workflow.run(normalized_question)
    finally:
        retriever.close()

    graph_builder = GraphBuilder(
        neo4j_uri=settings.neo4j_uri,
        neo4j_user=settings.neo4j_user,
        neo4j_password=settings.neo4j_password,
    )
    try:
        graph = graph_builder.get_query_subgraph(normalized_question)
    except Exception:
        graph = {"nodes": [], "edges": []}
    finally:
        graph_builder.close()

    graph = _annotate_graph_semantics(graph)

    answer = state.answer or "I could not find matching graph context for that question yet."
    insights = _compute_insights(state, graph, settings, query_router)
    return answer, state.citations, graph, insights


def _compute_insights(
    state: Any,
    graph: dict[str, list[dict[str, Any]]],
    settings: Settings,
    query_router: Any,
) -> dict[str, Any]:
    """Derive insight dicts from workflow state for the four hiring-manager panels."""
    # ── Routing ──────────────────────────────────────────────────────────────
    routing_rows: list[dict[str, Any]] = []
    for sub_q in state.sub_questions:
        if query_router is not None:
            d = query_router.decide(sub_q)
            routing_rows.append(
                {
                    "sub_question": sub_q,
                    "mode": d.mode,
                    "confidence": round(float(d.confidence), 3),
                    "reason": d.reason,
                }
            )
        else:
            routing_rows.append(
                {
                    "sub_question": sub_q,
                    "mode": settings.retrieval_mode,
                    "confidence": 1.0,
                    "reason": "static_config (adaptive routing disabled)",
                }
            )

    # ── Grounding ─────────────────────────────────────────────────────────────
    scores = [float(c.get("score", 0.0)) for c in state.citations]
    avg_score = round(sum(scores) / max(len(scores), 1), 4)
    all_entities: list[str] = []
    for c in state.citations:
        all_entities.extend(c.get("matched_entities", []) or [])
    unique_entities = list(dict.fromkeys(str(e) for e in all_entities if e))

    # ── Graph Evidence ────────────────────────────────────────────────────────
    all_neighbours: list[str] = []
    for c in state.citations:
        all_neighbours.extend(c.get("graph_neighbours", []) or [])
    unique_neighbours = list(dict.fromkeys(str(n) for n in all_neighbours if n))
    edge_semantics = _graph_edge_semantic_summary(graph)

    # ── Governance ────────────────────────────────────────────────────────────
    token_count = sum(len(sq.split()) for sq in state.sub_questions)
    estimated_cost = round(token_count * 0.000001, 8)

    return {
        "routing": routing_rows,
        "grounding": {
            "verified": state.verified,
            "citation_count": len(state.citations),
            "avg_score": avg_score,
            "unique_entities": unique_entities,
        },
        "graph_evidence": {
            "unique_entities": unique_entities,
            "unique_neighbours": unique_neighbours,
            "subgraph_nodes": len(graph.get("nodes", [])),
            "subgraph_edges": len(graph.get("edges", [])),
            "edge_semantics": edge_semantics,
        },
        "governance": {
            "model": settings.openai_model,
            "base_retrieval_mode": settings.retrieval_mode,
            "adaptive_routing_enabled": settings.adaptive_tool_routing_enabled,
            "vector_backend": settings.vector_store_backend,
            "sub_questions": list(state.sub_questions),
            "estimated_cost_usd": estimated_cost,
        },
        "contrastive": _load_contrastive_summary(),
    }


def _read_json_artifact(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_contrastive_summary() -> dict[str, Any]:
    contrastive = _read_json_artifact(_CONTRASTIVE_ARTIFACT_PATH)
    if contrastive is not None:
        metric_deltas = contrastive.get("metric_deltas", {})
        top_deltas: list[dict[str, float | str]] = []
        if isinstance(metric_deltas, dict):
            sortable: list[tuple[str, float]] = []
            for metric, value in metric_deltas.items():
                if isinstance(metric, str) and isinstance(value, int | float):
                    sortable.append((metric, float(value)))
            top_deltas = [
                {"metric": metric, "delta": round(value, 4)}
                for metric, value in sorted(sortable, key=lambda item: abs(item[1]), reverse=True)[:4]
            ]

        return {
            "source": "contrastive",
            "title": f"{contrastive.get('baseline_label', 'baseline')} vs {contrastive.get('candidate_label', 'candidate')}",
            "winner": str(contrastive.get("winner", "tie")),
            "improved_metrics": list(contrastive.get("improved_metrics", [])),
            "regressed_metrics": list(contrastive.get("regressed_metrics", [])),
            "top_deltas": top_deltas,
        }

    ablation = _read_json_artifact(_ABLATION_ARTIFACT_PATH)
    if ablation is not None:
        results = ablation.get("results", [])
        top_rows: list[dict[str, Any]] = []
        if isinstance(results, list):
            for row in results[:4]:
                if isinstance(row, dict):
                    top_rows.append(
                        {
                            "mode": str(row.get("mode", "")),
                            "context_recall": float(row.get("context_recall", 0.0)),
                            "context_precision": float(row.get("context_precision", 0.0)),
                        }
                    )
        return {
            "source": "ablation",
            "title": "Retrieval mode benchmark",
            "winner": str(ablation.get("winner", "unknown")),
            "results": top_rows,
        }

    return {}


def _annotate_graph_semantics(graph: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    edges = []
    for edge in graph.get("edges", []):
        relation = str(edge.get("type", "")).strip()
        edges.append({**edge, "semantic": describe_relationship_semantics(relation)})
    return {"nodes": list(graph.get("nodes", [])), "edges": edges}


def _graph_edge_semantic_summary(graph: dict[str, list[dict[str, Any]]]) -> list[dict[str, str]]:
    summaries: list[dict[str, str]] = []
    seen: set[str] = set()
    for edge in graph.get("edges", []):
        semantic = edge.get("semantic", {})
        if not isinstance(semantic, dict):
            continue
        relation = str(semantic.get("relation", "")).strip()
        if not relation or relation in seen:
            continue
        seen.add(relation)
        summaries.append(
            {
                "relation": relation,
                "predicate": str(semantic.get("predicate", "")),
                "domain": str(semantic.get("domain", "")),
                "range": str(semantic.get("range", "")),
            }
        )
        if len(summaries) >= 4:
            break
    return summaries


# ── Per-node-type fill colours ───────────────────────────────────────────────
_NODE_COLOURS: dict[str, str] = {
    "PortfolioMethod": "#7C3AED",
    "RiskMeasure": "#DC2626",
    "ConstraintType": "#D97706",
    "Estimator": "#059669",
    "AssetClass": "#0891B2",
    "FactorModel": "#2563EB",
    "MarketRegime": "#EA580C",
    "BenchmarkIndex": "#65A30D",
    "BacktestScenario": "#0D9488",
    "OptimizationProblem": "#7E22CE",
    "Solver": "#DB2777",
    "PlotType": "#E11D48",
    "ReportType": "#C2410C",
    "PythonFunction": "#0369A1",
    "PythonModule": "#0284C7",
    "PythonClass": "#4F46E5",
    "Parameter": "#9333EA",
    "Concept": "#6B7280",
    "Chunk": "#94A3B8",
    "DocPage": "#A1A1AA",
    "ExampleNotebook": "#78716C",
    "TestCase": "#64748B",
}
_DEFAULT_NODE_COLOUR = "#3B82F6"

# ── Per-relationship-type stroke colours ─────────────────────────────────────
_REL_COLOURS: dict[str, str] = {
    "IS_SUBTYPE_OF": "#DC2626",
    "ALTERNATIVE_TO": "#7C3AED",
    "SUPPORTS_RISK_MEASURE": "#059669",
    "USES_ESTIMATOR": "#0891B2",
    "HAS_PARAMETER": "#D97706",
    "HAS_CONSTRAINT": "#D97706",
    "REQUIRES": "#EA580C",
    "IMPLEMENTS": "#4F46E5",
    "DESCRIBES": "#0284C7",
    "DEMONSTRATES": "#0369A1",
    "VALIDATES": "#65A30D",
    "VALIDATED_AGAINST": "#65A30D",
    "RELATED_TO": "#64748B",
    "MENTIONS": "#94A3B8",
    "HAS_CHUNK": "#CBD5E1",
    "DECLARES": "#64748B",
    "PARAMETERIZED_BY": "#9333EA",
    "BENCHMARKED_ON": "#65A30D",
    "CALIBRATED_ON": "#0891B2",
    "PRECEDES": "#F59E0B",
}
_DEFAULT_REL_COLOUR = "#94A3B8"


def _render_graph_plot(graph: dict[str, list[dict[str, Any]]], height: int = 440) -> Any:
    """Render an interactive network graph using Plotly."""
    import networkx as nx
    import plotly.graph_objects as go

    bg = "#FFFFFF"
    fg = "#1E293B"
    muted = "#64748B"

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    if not nodes:
        fig = go.Figure()
        fig.add_annotation(
            text="Ask a question to populate the knowledge graph.",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14, "color": muted},
        )
        fig.update_layout(
            height=height,
            paper_bgcolor=bg,
            plot_bgcolor=bg,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            xaxis={"visible": False},
            yaxis={"visible": False},
        )
        return fig

    excluded_types = {"Chunk", "DocPage", "ExampleNotebook", "TestCase"}
    priority_types = {"PortfolioMethod", "RiskMeasure", "ConstraintType", "Estimator", "PythonClass", "Concept"}

    filtered_nodes = [n for n in nodes if str((n.get("labels") or ["Concept"])[0]) not in excluded_types]
    if len(filtered_nodes) > 35:

        def _score(n: dict[str, Any]) -> tuple[bool, int]:
            t = str((n.get("labels") or ["Concept"])[0])
            deg = sum(1 for e in edges if str(e.get("source")) == str(n.get("id")) or str(e.get("target")) == str(n.get("id")))
            return (t in priority_types, deg)

        filtered_nodes = sorted(filtered_nodes, key=_score, reverse=True)[:35]

    kept_ids = {str(n.get("id", "")) for n in filtered_nodes}

    graph_nx = nx.DiGraph()
    node_meta: dict[str, dict[str, str]] = {}
    for node in filtered_nodes:
        nid = str(node.get("id", ""))
        name = str(node.get("name", "")).strip() or "Unnamed"
        labels = node.get("labels", [])
        node_type = str(labels[0]) if labels else "Concept"
        graph_nx.add_node(nid)
        node_meta[nid] = {"name": name, "type": node_type}

    skip_rel = {"HAS_CHUNK", "DECLARES"}
    for edge in edges:
        src = str(edge.get("source", ""))
        tgt = str(edge.get("target", ""))
        if src == tgt or src not in kept_ids or tgt not in kept_ids:
            continue
        rel = str(edge.get("type", ""))
        graph_nx.add_edge(src, tgt, rel=rel, colour=_REL_COLOURS.get(rel, _DEFAULT_REL_COLOUR), show_label=rel not in skip_rel)

    if graph_nx.number_of_nodes() == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No connected nodes found for this query after filtering.",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14, "color": muted},
        )
        fig.update_layout(
            height=height,
            paper_bgcolor=bg,
            plot_bgcolor=bg,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            xaxis={"visible": False},
            yaxis={"visible": False},
        )
        return fig

    node_count = graph_nx.number_of_nodes()
    edge_count = graph_nx.number_of_edges()
    density = edge_count / max(node_count, 1)

    # In dense graphs, prune low-signal edge types to reduce visual hairballs.
    low_signal_rel = {"RELATED_TO", "MENTIONS", "DESCRIBES", "DEMONSTRATES"}
    if density > 4:
        low_signal_rel.update({"VALIDATES", "VALIDATED_AGAINST"})
    if density > 2.5:
        to_remove = [(u, v) for u, v, d in graph_nx.edges(data=True) if str(d.get("rel", "")) in low_signal_rel]
        graph_nx.remove_edges_from(to_remove)

    isolates = list(nx.isolates(graph_nx))
    graph_nx.remove_nodes_from(isolates)
    if graph_nx.number_of_nodes() == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No connected nodes found for this query after filtering.",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14, "color": muted},
        )
        fig.update_layout(
            height=height,
            paper_bgcolor=bg,
            plot_bgcolor=bg,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            xaxis={"visible": False},
            yaxis={"visible": False},
        )
        return fig

    node_count = graph_nx.number_of_nodes()
    edge_count = graph_nx.number_of_edges()
    density = edge_count / max(node_count, 1)

    # Spread dense graphs more aggressively to reduce node/label collisions.
    if density > 4:
        k_value = 3.0 / max(node_count**0.5, 1)
        iterations = 550
    elif density > 2:
        k_value = 2.4 / max(node_count**0.5, 1)
        iterations = 430
    else:
        k_value = 1.8 / max(node_count**0.5, 1)
        iterations = 320
    pos = nx.spring_layout(graph_nx, seed=7, k=k_value, iterations=iterations)

    # Simple deterministic collision-repulsion pass to enforce minimum spacing.
    node_ids = list(graph_nx.nodes())
    min_sep = 0.11 if node_count >= 30 else 0.085
    for _ in range(12):
        moved = False
        for i, a in enumerate(node_ids):
            xa, ya = pos[a]
            for b in node_ids[i + 1 :]:
                xb, yb = pos[b]
                dx = xb - xa
                dy = yb - ya
                dist2 = dx * dx + dy * dy
                if dist2 == 0:
                    dx = 1e-4
                    dy = -1e-4
                    dist2 = dx * dx + dy * dy
                dist = dist2**0.5
                if dist < min_sep:
                    push = (min_sep - dist) / 2.0
                    ux = dx / dist
                    uy = dy / dist
                    pos[a] = (xa - ux * push, ya - uy * push)
                    pos[b] = (xb + ux * push, yb + uy * push)
                    xa, ya = pos[a]
                    moved = True
        if not moved:
            break

    high_signal_rel = {
        "IS_SUBTYPE_OF",
        "ALTERNATIVE_TO",
        "SUPPORTS_RISK_MEASURE",
        "USES_ESTIMATOR",
        "HAS_PARAMETER",
        "HAS_CONSTRAINT",
        "REQUIRES",
        "IMPLEMENTS",
        "PARAMETERIZED_BY",
        "BENCHMARKED_ON",
        "CALIBRATED_ON",
        "PRECEDES",
    }

    edge_x_hi: list[float | None] = []
    edge_y_hi: list[float | None] = []
    edge_x_lo: list[float | None] = []
    edge_y_lo: list[float | None] = []
    edge_hover_x: list[float] = []
    edge_hover_y: list[float] = []
    edge_hover_text: list[str] = []

    for source, target, data in graph_nx.edges(data=True):
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        rel = str(data.get("rel", ""))
        if rel in high_signal_rel:
            edge_x_hi.extend([x0, x1, None])
            edge_y_hi.extend([y0, y1, None])
        else:
            edge_x_lo.extend([x0, x1, None])
            edge_y_lo.extend([y0, y1, None])

        # Midpoint hover targets for relationship labels without drawing edge text.
        mx = (x0 + x1) / 2.0
        my = (y0 + y1) / 2.0
        dx = x1 - x0
        dy = y1 - y0
        norm = (dx * dx + dy * dy) ** 0.5
        if norm > 0:
            mx += (-dy / norm) * 0.012
            my += (dx / norm) * 0.012
        edge_hover_x.append(mx)
        edge_hover_y.append(my)
        semantic = data.get("semantic", {}) if isinstance(data.get("semantic", {}), dict) else {}
        predicate = html.escape(str(semantic.get("predicate", "")))
        domain = html.escape(str(semantic.get("domain", "")))
        range_name = html.escape(str(semantic.get("range", "")))
        semantic_line = ""
        if predicate:
            semantic_line = f"<br>{predicate}"
            if domain or range_name:
                semantic_line += f"<br>{domain or 'owl:Thing'} → {range_name or 'owl:Thing'}"
        source_name = html.escape(node_meta[source]["name"])
        target_name = html.escape(node_meta[target]["name"])
        edge_hover_text.append(f"{html.escape(rel)}<br>{source_name} -> {target_name}{semantic_line}")

    edge_trace_lo = go.Scatter(
        x=edge_x_lo,
        y=edge_y_lo,
        mode="lines",
        line={"width": 0.9, "color": "rgba(100,116,139,0.25)", "shape": "spline", "smoothing": 0.25},
        hoverinfo="skip",
        showlegend=False,
    )

    edge_trace_hi = go.Scatter(
        x=edge_x_hi,
        y=edge_y_hi,
        mode="lines",
        line={"width": 1.5, "color": "rgba(71,85,105,0.55)", "shape": "spline", "smoothing": 0.25},
        hoverinfo="skip",
        showlegend=False,
    )

    edge_hover_trace = go.Scatter(
        x=edge_hover_x,
        y=edge_hover_y,
        mode="markers",
        marker={"size": 10, "color": "rgba(0,0,0,0)"},
        hovertext=edge_hover_text,
        hoverinfo="text",
        showlegend=False,
    )

    node_degree = dict(graph_nx.degree())
    type_to_nodes: dict[str, list[str]] = {}
    for node_id in graph_nx.nodes():
        node_type = node_meta[node_id]["type"]
        type_to_nodes.setdefault(node_type, []).append(node_id)

    # Label only the most informative nodes on dense graphs; others remain interactive via hover.
    label_budget = 16 if node_count >= 30 else 24
    ranked_for_labels = sorted(graph_nx.nodes(), key=lambda n: node_degree.get(n, 0), reverse=True)[:label_budget]
    label_nodes = set(ranked_for_labels)

    node_traces: list[Any] = []
    for node_type in sorted(type_to_nodes.keys()):
        ids = type_to_nodes[node_type]
        x_vals = [pos[i][0] for i in ids]
        y_vals = [pos[i][1] for i in ids]
        hover_text = [
            f"{html.escape(node_meta[i]['name'])}<br>Type: {html.escape(node_type)}<br>Degree: {node_degree.get(i, 0)}"
            for i in ids
        ]
        labels = [
            (node_meta[i]["name"][:20] + ("..." if len(node_meta[i]["name"]) > 20 else "")) if i in label_nodes else ""
            for i in ids
        ]
        sizes = [14 + min(node_degree.get(i, 0) * 2, 16) for i in ids]

        node_traces.append(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers+text",
                text=labels,
                textposition="top center",
                textfont={"size": 10, "color": fg},
                hovertext=hover_text,
                hoverinfo="text",
                marker={
                    "size": sizes,
                    "color": _NODE_COLOURS.get(node_type, _DEFAULT_NODE_COLOUR),
                    "line": {"width": 1.5, "color": "#FFFFFF"},
                    "opacity": 0.95,
                },
                name=node_type,
            )
        )

    fig = go.Figure(data=[edge_trace_lo, edge_trace_hi, edge_hover_trace, *node_traces])
    fig.update_layout(
        height=height,
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        xaxis={"visible": False},
        yaxis={"visible": False},
        showlegend=True,
        font={"color": fg},
        legend={
            "bgcolor": "rgba(255,255,255,0.9)",
            "bordercolor": "#CBD5E1",
            "borderwidth": 1,
            "font": {"size": 10, "color": fg},
            "x": 0.99,
            "y": 0.01,
            "xanchor": "right",
            "yanchor": "bottom",
        },
        dragmode="pan",
    )
    return fig


def _render_graph_image(graph: dict[str, list[dict[str, Any]]]) -> Any:
    """Render the knowledge graph as a matplotlib image via networkx (fallback)."""
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import networkx as nx
    from PIL import Image as PILImage

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    if not nodes:
        return None

    _EXCLUDED_TYPES = {"Chunk", "DocPage", "ExampleNotebook", "TestCase"}
    # Priority types to always keep; others trimmed if graph is large
    _PRIORITY_TYPES = {"PortfolioMethod", "RiskMeasure", "ConstraintType", "Estimator", "PythonClass", "Concept"}

    G = nx.DiGraph()
    node_labels: dict[str, str] = {}
    node_types: dict[str, str] = {}

    for node in nodes:
        nid = str(node.get("id", ""))
        name = str(node.get("name", "")).strip() or "Unnamed"
        labels = node.get("labels", [])
        primary = str(labels[0]) if labels else "Concept"
        if primary in _EXCLUDED_TYPES:
            continue
        G.add_node(nid)
        node_labels[nid] = name
        node_types[nid] = primary

    for edge in edges:
        src = str(edge.get("source", ""))
        tgt = str(edge.get("target", ""))
        rel = str(edge.get("type", ""))
        if src != tgt and G.has_node(src) and G.has_node(tgt):
            G.add_edge(src, tgt, label=rel, semantic=edge.get("semantic", {}))

    # Remove isolated nodes to reduce clutter
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    # Cap at 25 nodes: keep highest-degree first, priority types preferred
    if len(G.nodes()) > 25:
        scored = sorted(
            G.nodes(),
            key=lambda n: (node_types.get(n, "") in _PRIORITY_TYPES, G.degree(n)),
            reverse=True,
        )
        keep = set(scored[:25])
        remove = [n for n in list(G.nodes()) if n not in keep]
        G.remove_nodes_from(remove)

    if not G.nodes():
        return None

    n_nodes = len(G.nodes())
    # Spring layout with high k spreads nodes far apart; more iterations = more stable
    pos = nx.spring_layout(G, k=5.0 / max(n_nodes**0.4, 1), seed=7, iterations=300)

    node_list = list(G.nodes())
    node_colours = [_NODE_COLOURS.get(node_types.get(n, "Concept"), _DEFAULT_NODE_COLOUR) for n in node_list]

    # Node sizes: hubs slightly larger but keep variance low to avoid crowding
    degree = dict(G.degree())
    node_sizes = [180 + degree.get(n, 0) * 40 for n in node_list]

    fig, ax = plt.subplots(figsize=(14, 9), facecolor="#0F172A")
    ax.set_facecolor("#0F172A")
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.10)

    # Draw edges first (behind nodes)
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color="#334155",
        arrows=True,
        arrowsize=10,
        width=0.8,
        alpha=0.6,
        connectionstyle="arc3,rad=0.08",
        node_size=node_sizes,
        min_source_margin=12,
        min_target_margin=12,
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=node_list,
        ax=ax,
        node_color=node_colours,
        node_size=node_sizes,
        alpha=0.92,
        linewidths=1.0,
        edgecolors="#0F172A",
    )

    # Labels
    short_labels = {
        n: (node_labels.get(n, n)[:16] + "…" if len(node_labels.get(n, n)) > 16 else node_labels.get(n, n)) for n in node_list
    }
    nx.draw_networkx_labels(G, pos, labels=short_labels, ax=ax, font_size=7, font_color="#F1F5F9", font_weight="bold")

    # Only draw edge labels for the most meaningful relationship types
    _SHOW_REL_LABELS = {"IS_SUBTYPE_OF", "ALTERNATIVE_TO", "SUPPORTS_RISK_MEASURE", "IMPLEMENTS", "USES_ESTIMATOR"}
    edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True) if d.get("label", "") in _SHOW_REL_LABELS}
    if edge_labels:
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            ax=ax,
            font_size=6,
            font_color="#94A3B8",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "#1E293B", "edgecolor": "none", "alpha": 0.7},
        )

    # Title
    ax.set_title("Knowledge Graph", color="#94A3B8", fontsize=11, pad=8, loc="left")

    # Legend
    seen: dict[str, str] = {}
    for nid in G.nodes():
        pt = node_types.get(nid, "Concept")
        if pt not in seen:
            seen[pt] = _NODE_COLOURS.get(pt, _DEFAULT_NODE_COLOUR)
    patches = [mpatches.Patch(color=c, label=t) for t, c in sorted(seen.items())]
    ax.legend(
        handles=patches,
        loc="lower center",
        fontsize=7,
        framealpha=0.6,
        ncol=min(len(patches), 6),
        facecolor="#1E293B",
        edgecolor="#334155",
        labelcolor="#CBD5E1",
        bbox_to_anchor=(0.5, -0.02),
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return PILImage.open(buf).copy()


# ── Insight-panel rendering helpers ─────────────────────────────────────────

_MODE_COLOURS: dict[str, str] = {
    "dense": "#3B82F6",
    "sparse": "#8B5CF6",
    "graph": "#10B981",
    "hybrid_rerank": "#F59E0B",
}

_EMPTY_ROUTING_HTML = (
    "<p style='color:#9CA3AF;font-size:13px;padding:8px'>"
    "Ask a question above to see how the AI decides which search tool to use for each part of your query."
    "</p>"
)
_EMPTY_GROUNDING_HTML = (
    "<p style='color:#9CA3AF;font-size:13px;padding:8px'>"
    "Ask a question above to see how well the answer is backed by evidence "
    "(versus the AI guessing)."
    "</p>"
)
_EMPTY_GRAPH_EVIDENCE_HTML = (
    "<p style='color:#9CA3AF;font-size:13px;padding:8px'>"
    "Ask a question above to see which concepts and relationships from the knowledge graph "
    "were used to construct the answer."
    "</p>"
)
_EMPTY_GOVERNANCE_HTML = (
    "<p style='color:#9CA3AF;font-size:13px;padding:8px'>"
    "Ask a question above to see which AI model was used, what safety guardrails were active, "
    "and how much the query cost in tokens."
    "</p>"
)
_EMPTY_CONTRASTIVE_HTML = (
    "<p style='color:#9CA3AF;font-size:13px;padding:8px'>"
    "No contrastive artifact detected yet. Add <code>artifacts/eval/contrastive.json</code> "
    "to show baseline-vs-candidate summaries here, or rely on the ablation benchmark fallback."
    "</p>"
)


def _render_graph_svg(graph: dict[str, list[dict[str, Any]]], width: int = 800, height: int = 400) -> str:
    """Render a simple SVG representation of the knowledge graph."""
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    if not nodes:
        return (
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>"
            f"<text x='{width // 2}' y='{height // 2}' text-anchor='middle' "
            "fill='#9CA3AF' font-size='14' font-family='sans-serif'>"
            "No graph data available"
            "</text></svg>"
        )

    import math

    node_positions: dict[str, tuple[float, float]] = {}
    n = len(nodes)
    cx, cy, r = width / 2, height / 2, min(width, height) * 0.35
    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / n
        node_positions[str(node.get("id", i))] = (
            cx + r * math.cos(angle),
            cy + r * math.sin(angle),
        )

    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>"]

    for edge in edges:
        src = str(edge.get("source", ""))
        tgt = str(edge.get("target", ""))
        etype = html.escape(str(edge.get("type", "")))
        if src in node_positions and tgt in node_positions:
            x1, y1 = node_positions[src]
            x2, y2 = node_positions[tgt]
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            semantic = edge.get("semantic", {}) if isinstance(edge.get("semantic", {}), dict) else {}
            tooltip = html.escape(
                f"{etype} | {semantic.get('predicate', '')} | {semantic.get('domain', '')} -> {semantic.get('range', '')}"
            )
            parts.append(
                (
                    f"<g><title>{tooltip}</title>"
                    f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' "
                    "stroke='#94A3B8' stroke-width='1.5'/></g>"
                )
            )
            if etype:
                parts.append(
                    f"<text x='{mx:.1f}' y='{my:.1f}' text-anchor='middle' "
                    "fill='#64748B' font-size='10' font-family='sans-serif'>"
                    f"{etype}</text>"
                )

    for node in nodes:
        nid = str(node.get("id", ""))
        name = html.escape(str(node.get("name", "")).strip() or "Unnamed")
        labels = node.get("labels", [])
        primary = str(labels[0]) if labels else "Concept"
        colour = _NODE_COLOURS.get(primary, _DEFAULT_NODE_COLOUR)
        if nid in node_positions:
            x, y = node_positions[nid]
            parts.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='20' fill='{colour}' stroke='white' stroke-width='2'/>")
            parts.append(
                f"<text x='{x:.1f}' y='{y + 32:.1f}' text-anchor='middle' "
                f"fill='#1E293B' font-size='11' font-family='sans-serif'>{name}</text>"
            )

    parts.append("</svg>")
    return "".join(parts)


def _badge(text: str, colour: str) -> str:
    safe = html.escape(str(text))
    return (
        f"<span style='display:inline-block;padding:2px 8px;border-radius:12px;"
        f"background:{colour};color:white;font-size:11px;font-weight:600'>{safe}</span>"
    )


def _render_routing_html(insights: dict[str, Any]) -> str:
    """Routing Rationale panel – adaptive tool selection per sub-question."""
    rows = insights.get("routing", [])
    if not rows:
        return _EMPTY_ROUTING_HTML

    header = (
        "<table style='width:100%;border-collapse:collapse;font-size:13px'>"
        "<thead><tr style='background:#F1F5F9'>"
        "<th style='padding:6px 10px;text-align:left;border-bottom:1px solid #E2E8F0'>Sub-question</th>"
        "<th style='padding:6px 10px;text-align:center;border-bottom:1px solid #E2E8F0'>Tool selected</th>"
        "<th style='padding:6px 10px;text-align:center;border-bottom:1px solid #E2E8F0'>Confidence</th>"
        "<th style='padding:6px 10px;text-align:left;border-bottom:1px solid #E2E8F0'>Reason</th>"
        "</tr></thead><tbody>"
    )
    body_rows: list[str] = []
    for index, row in enumerate(rows):
        mode = str(row.get("mode", "hybrid_rerank"))
        colour = _MODE_COLOURS.get(mode, "#6B7280")
        conf = float(row.get("confidence", 0.0))
        pct = min(100, int(conf * 100))
        conf_bar = (
            f"<div style='background:#E2E8F0;border-radius:4px;height:8px;width:100px;"
            f"display:inline-block;vertical-align:middle'>"
            f"<div style='background:{colour};border-radius:4px;height:8px;width:{pct}%'></div></div>"
            f"&nbsp;<span style='font-size:11px;color:#64748B'>{conf:.2f}</span>"
        )
        bg = "#FFFFFF" if index % 2 == 0 else "#F8FAFC"
        body_rows.append(
            f"<tr style='background:{bg}'>"
            f"<td style='padding:6px 10px;color:#334155'>{html.escape(str(row.get('sub_question', '')))}</td>"
            f"<td style='padding:6px 10px;text-align:center'>{_badge(mode, colour)}</td>"
            f"<td style='padding:6px 10px'>{conf_bar}</td>"
            f"<td style='padding:6px 10px;color:#64748B;font-style:italic'>{html.escape(str(row.get('reason', '')))}</td>"
            f"</tr>"
        )
    return f"<div style='overflow-x:auto'>{header}{''.join(body_rows)}</tbody></table></div>"


def _render_grounding_html(insights: dict[str, Any]) -> str:
    """Grounding & Faithfulness panel – verification flag, scores, entity coverage."""
    g = insights.get("grounding", {})
    if not g:
        return _EMPTY_GROUNDING_HTML

    verified = bool(g.get("verified", False))
    citation_count = int(g.get("citation_count", 0))
    avg_score = float(g.get("avg_score", 0.0))
    entities = list(g.get("unique_entities", []))

    verdict_colour = "#10B981" if verified else "#EF4444"
    verdict_icon = "✔ Verified" if verified else "✘ Unverified"
    verdict_note = "Token-overlap ≥ 25 % and citations present." if verified else "Token-overlap below threshold or no citations."

    entity_chips = "".join(_badge(e, "#6366F1") + "&nbsp;" for e in entities[:12])
    overflow = (
        f"<span style='color:#6B7280;font-size:11px'>&hellip;&nbsp;{len(entities) - 12} more</span>" if len(entities) > 12 else ""
    )
    pct = min(100, int(avg_score * 100))
    score_bar = (
        f"<div style='background:#E2E8F0;border-radius:4px;height:10px;width:180px;"
        f"display:inline-block;vertical-align:middle'>"
        f"<div style='background:#3B82F6;border-radius:4px;height:10px;width:{pct}%'></div></div>"
        f"&nbsp;<span style='font-size:12px;color:#475569'>{avg_score:.4f}</span>"
    )
    table_rows = [
        (
            "Answer verification",
            f"<span style='color:{verdict_colour};font-weight:700'>{verdict_icon}</span>"
            f"&nbsp;&mdash;&nbsp;<span style='color:#6B7280;font-size:12px'>{html.escape(verdict_note)}</span>",
        ),
        ("Citations retrieved", f"<strong>{citation_count}</strong>"),
        ("Avg retrieval score", score_bar),
        ("Matched entities", (entity_chips + overflow) if entities else "<span style='color:#9CA3AF'>none</span>"),
    ]
    cells = "".join(
        f"<tr><td style='padding:7px 12px;color:#64748B;font-size:13px;white-space:nowrap'>{label}</td>"
        f"<td style='padding:7px 12px;font-size:13px'>{value}</td></tr>"
        for label, value in table_rows
    )
    return f"<table style='border-collapse:collapse;width:100%'>{cells}</table>"


def _render_graph_evidence_html(insights: dict[str, Any]) -> str:
    """Graph Evidence panel – entity nodes, neighbourhood traversal, subgraph counts."""
    ge = insights.get("graph_evidence", {})
    if not ge:
        return _EMPTY_GRAPH_EVIDENCE_HTML

    entities = list(ge.get("unique_entities", []))
    neighbours = list(ge.get("unique_neighbours", []))
    subgraph_nodes = int(ge.get("subgraph_nodes", 0))
    subgraph_edges = int(ge.get("subgraph_edges", 0))
    edge_semantics = list(ge.get("edge_semantics", []))

    entity_chips = "".join(_badge(e, "#10B981") + "&nbsp;" for e in entities[:15])
    neighbour_chips = "".join(_badge(n, "#8B5CF6") + "&nbsp;" for n in neighbours[:10])
    e_overflow = (
        f"<span style='color:#6B7280;font-size:11px'>&hellip;&nbsp;{len(entities) - 15} more</span>" if len(entities) > 15 else ""
    )
    n_overflow = (
        f"<span style='color:#6B7280;font-size:11px'>&hellip;&nbsp;{len(neighbours) - 10} more</span>"
        if len(neighbours) > 10
        else ""
    )
    semantic_rows: list[str] = []
    for item in edge_semantics[:4]:
        if not isinstance(item, dict):
            continue
        relation = html.escape(str(item.get("relation", "")))
        predicate = html.escape(str(item.get("predicate", "")))
        domain = html.escape(str(item.get("domain", "owl:Thing")))
        range_name = html.escape(str(item.get("range", "owl:Thing")))
        semantic_rows.append(
            f"<div style='margin:2px 0'>{_badge(relation, '#475569')} "
            f"<span style='color:#475569'>{predicate} · {domain} → {range_name}</span></div>"
        )
    semantic_block = "".join(semantic_rows) if semantic_rows else "<span style='color:#9CA3AF'>no semantic edge details</span>"
    table_rows = [
        ("Subgraph nodes (visualised)", f"<strong>{subgraph_nodes}</strong>"),
        ("Subgraph edges (visualised)", f"<strong>{subgraph_edges}</strong>"),
        (
            "Matched domain entities",
            (entity_chips + e_overflow)
            if entities
            else "<span style='color:#9CA3AF'>none yet &ndash; graph not populated</span>",
        ),
        (
            "1-hop graph neighbours",
            (neighbour_chips + n_overflow)
            if neighbours
            else "<span style='color:#9CA3AF'>none yet &ndash; graph not populated</span>",
        ),
        ("OWL/RDF edge semantics", semantic_block),
    ]
    cells = "".join(
        f"<tr><td style='padding:7px 12px;color:#64748B;font-size:13px;white-space:nowrap;vertical-align:top'>{label}</td>"
        f"<td style='padding:7px 12px;font-size:13px'>{value}</td></tr>"
        for label, value in table_rows
    )
    return f"<table style='border-collapse:collapse;width:100%'>{cells}</table>"


def _render_governance_html(insights: dict[str, Any]) -> str:
    """Governance & Cost panel – model, retrieval mode, cost controls, safe tool use."""
    gov = insights.get("governance", {})
    if not gov:
        return _EMPTY_GOVERNANCE_HTML

    model = html.escape(str(gov.get("model", "\u2014")))
    base_mode = str(gov.get("base_retrieval_mode", "\u2014"))
    adaptive = bool(gov.get("adaptive_routing_enabled", False))
    backend = html.escape(str(gov.get("vector_backend", "\u2014")))
    sub_questions = list(gov.get("sub_questions", []))
    cost = float(gov.get("estimated_cost_usd", 0.0))

    mode_badge = _badge(base_mode, _MODE_COLOURS.get(base_mode, "#6B7280"))
    adaptive_badge = _badge("ON \u2714" if adaptive else "OFF", "#10B981" if adaptive else "#EF4444")
    guard_badge = _badge("ALLOWLISTED TEMPLATES \u2714", "#10B981")

    sq_items = "".join(f"<li style='margin:2px 0;font-size:12px;color:#334155'>{html.escape(q)}</li>" for q in sub_questions)
    sq_block = (
        f"<ol style='margin:4px 0 0 16px;padding:0'>{sq_items}</ol>" if sq_items else "<span style='color:#9CA3AF'>\u2014</span>"
    )

    table_rows = [
        ("LLM model", f"<code style='background:#F1F5F9;padding:1px 5px;border-radius:3px'>{model}</code>"),
        ("Base retrieval mode", mode_badge),
        ("Adaptive tool routing", adaptive_badge),
        ("Vector backend", f"<code style='background:#F1F5F9;padding:1px 5px;border-radius:3px'>{backend}</code>"),
        ("NL\u2192Cypher guardrails", guard_badge),
        ("Estimated token cost", f"<span style='color:#475569'>${cost:.8f}</span>"),
        ("Agent sub-questions", sq_block),
    ]
    cells = "".join(
        f"<tr><td style='padding:7px 12px;color:#64748B;font-size:13px;white-space:nowrap;vertical-align:top'>{label}</td>"
        f"<td style='padding:7px 12px;font-size:13px'>{value}</td></tr>"
        for label, value in table_rows
    )
    return f"<table style='border-collapse:collapse;width:100%'>{cells}</table>"


def _render_contrastive_html(insights: dict[str, Any]) -> str:
    summary = insights.get("contrastive", {})
    if not summary:
        return _EMPTY_CONTRASTIVE_HTML

    source = str(summary.get("source", ""))
    title = html.escape(str(summary.get("title", "Benchmark summary")))
    winner = html.escape(str(summary.get("winner", "unknown")))

    if source == "contrastive":
        improved = [str(item) for item in summary.get("improved_metrics", [])[:6]]
        regressed = [str(item) for item in summary.get("regressed_metrics", [])[:6]]
        top_deltas = list(summary.get("top_deltas", []))
        delta_rows: list[str] = []
        for item in top_deltas:
            if not isinstance(item, dict):
                continue
            metric = html.escape(str(item.get("metric", "")))
            delta = float(item.get("delta", 0.0))
            colour = "#10B981" if delta >= 0 else "#EF4444"
            delta_rows.append(
                (f"<div style='margin:2px 0'><strong>{metric}</strong>: <span style='color:{colour}'>{delta:+.4f}</span></div>")
            )
        empty_badge_text = "<span style='color:#9CA3AF'>none</span>"
        improved_badges = "".join(_badge(metric, "#10B981") + "&nbsp;" for metric in improved) or empty_badge_text
        regressed_badges = "".join(_badge(metric, "#EF4444") + "&nbsp;" for metric in regressed) or empty_badge_text
        delta_block = "".join(delta_rows) if delta_rows else "<span style='color:#9CA3AF'>no deltas recorded</span>"
        return "".join(
            [
                f"<div><div style='font-weight:700;color:#1E293B;margin-bottom:8px'>{title}</div>",
                f"<div style='margin-bottom:8px'>Winner: {_badge(winner, '#3B82F6')}</div>",
                "<div style='margin-bottom:8px'>"
                "<div style='color:#64748B;font-size:12px;margin-bottom:4px'>Improved metrics</div>"
                f"{improved_badges}</div>",
                "<div style='margin-bottom:8px'>"
                "<div style='color:#64748B;font-size:12px;margin-bottom:4px'>Regressed metrics</div>"
                f"{regressed_badges}</div>",
                f"<div style='color:#64748B;font-size:12px;margin-bottom:4px'>Largest deltas</div>{delta_block}</div>",
            ]
        )

    results = list(summary.get("results", []))
    rows: list[str] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        mode = str(item.get("mode", ""))
        rows.append(
            f"<tr><td style='padding:6px 10px'>{_badge(mode, _MODE_COLOURS.get(mode, '#6B7280'))}</td>"
            f"<td style='padding:6px 10px'>{float(item.get('context_recall', 0.0)):.4f}</td>"
            f"<td style='padding:6px 10px'>{float(item.get('context_precision', 0.0)):.4f}</td></tr>"
        )
    return "".join(
        [
            f"<div><div style='font-weight:700;color:#1E293B;margin-bottom:8px'>{title}</div>",
            f"<div style='margin-bottom:8px'>Current winner: {_badge(winner, '#3B82F6')}</div>",
            "<table style='width:100%;border-collapse:collapse;font-size:13px'>",
            "<thead><tr style='background:#F1F5F9'><th style='padding:6px 10px;text-align:left'>Mode</th>",
            "<th style='padding:6px 10px;text-align:left'>Recall</th>",
            "<th style='padding:6px 10px;text-align:left'>Precision</th></tr></thead>",
            f"<tbody>{''.join(rows)}</tbody></table></div>",
        ]
    )


# ─────────────────────────────────────────────────────────────────────────────

_LOADING_HTML = "<p style='color:#6B7280;font-size:13px;padding:8px'>⏳ Searching knowledge graph…</p>"


def _format_answer_markdown(answer: str, citations: list) -> str:
    """Bold entity names from citations when the answer is plain prose."""
    if not answer:
        return answer
    # If already markdown-formatted, trust it as-is
    if any(c in answer for c in ("**", "##", "\n- ", "\n* ", "\n1.")):
        return answer
    entities: set[str] = set()
    for c in citations:
        for e in c.get("matched_entities") or []:
            e_str = str(e).strip()
            if len(e_str) > 3:
                entities.add(e_str)
    formatted = answer
    for entity in sorted(entities, key=len, reverse=True):
        if entity in formatted:
            formatted = formatted.replace(entity, f"**{entity}**", 1)
    return formatted


def _render_summary_card(insights: dict, citations: list, top_k: int = 10) -> str:
    """'What just happened' strip shown below the chatbot."""
    grounding = insights.get("grounding", {})
    graph_ev = insights.get("graph_evidence", {})
    gov = insights.get("governance", {})
    n_found = len(citations)
    avg_score = grounding.get("avg_score", 0.0)
    verified = grounding.get("verified", False)
    subgraph_nodes = graph_ev.get("subgraph_nodes", 0)
    cost = gov.get("estimated_cost_usd", 0.0)
    v_icon = "\u2714 Verified" if verified else "\u2718 Not verified"
    v_colour = "#10B981" if verified else "#EF4444"
    return (
        "<div style='background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;"
        "padding:8px 16px;font-size:12px;color:#475569;display:flex;gap:20px;"
        "flex-wrap:wrap;margin-top:4px'>"
        f"<span>\U0001f4da <strong>{n_found}</strong> / {top_k} sources</span>"
        f"<span>\U0001f578 <strong>{subgraph_nodes}</strong> graph nodes</span>"
        f"<span>\U0001f4ca avg score <strong>{avg_score:.3f}</strong></span>"
        f"<span style='color:{v_colour}'><strong>{v_icon}</strong></span>"
        f"<span>\U0001f4b0 est. cost <strong>${cost:.6f}</strong></span>"
        "</div>"
    )


def create_gradio_app(
    top_k_default: int = 10,
    graph_max_nodes: int = 40,
    graph_max_edges: int = 80,
):
    def _handle_submit(
        question: str,
        history: list[dict[str, str]] | None,
    ):
        """Generator: first yield shows loading state, second yield shows results."""
        top_k = 15
        normalized = question.strip()
        if not normalized:
            return

        # ── Loading state – show immediately ──────────────────────────────
        loading_history = list(history or []) + [
            {"role": "user", "content": normalized},
            {"role": "assistant", "content": "⏳ Searching the knowledge graph…"},
        ]
        yield (
            gr.update(),
            loading_history,
            _LOADING_HTML,
            _LOADING_HTML,
            [],
            _LOADING_HTML,
            _render_graph_plot({"nodes": [], "edges": []}),
            _LOADING_HTML,
            _LOADING_HTML,
            gr.update(),
        )

        # ── Run the full pipeline ─────────────────────────────────────────
        answer, citations, graph, insights = run_query_with_graph(
            normalized,
            top_k=top_k,
            graph_max_nodes=graph_max_nodes,
            graph_max_edges=graph_max_edges,
        )
        formatted_answer = _format_answer_markdown(answer, citations)
        final_history = list(history or []) + [
            {"role": "user", "content": normalized},
            {"role": "assistant", "content": formatted_answer},
        ]
        yield (
            "",
            final_history,
            _render_routing_html(insights),
            _render_grounding_html(insights),
            citations,
            _render_graph_evidence_html(insights),
            _render_graph_plot(graph),
            _render_governance_html(insights),
            _render_contrastive_html(insights),
            gr.update(selected=0),
        )

    with gr.Blocks(
        title="Portfolio AI Assistant — Knowledge Graph + RAG",
        theme=gr.themes.Soft(),
        css=(
            "@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');"
            "* { font-family: 'Roboto', sans-serif !important; }"
            "footer {display:none !important}"
            ":root { color-scheme: light !important; }"
            "html, body, .gradio-container { background:#F8FAFC !important; color:#1E293B !important; }"
            ".gradio-container, .gradio-container * { color: #1E293B !important; }"
            ".gradio-container .block, .gradio-container .panel, .gradio-container .wrap, .gradio-container .prose {"
            " background:#FFFFFF !important; color:#1E293B !important; }"
            ".gradio-container input, .gradio-container textarea {"
            " background:#FFFFFF !important; color:#1E293B !important; border-color:#CBD5E1 !important; }"
            "#query-row { gap: 10px; }"
            "#graph-panel { min-height: 300px; }"
            "#insight-tabs [role='tablist'] { background: transparent !important; }"
            "#insight-tabs [role='tablist'] button { white-space: normal !important;"
            " line-height: 1.2 !important; }"
            "#insight-tabs [role='tablist'] button {"
            " color:#334155 !important; background:#EEF2F7 !important; border:1px solid #CBD5E1 !important;"
            " opacity:1 !important; }"
            "#insight-tabs [role='tablist'] button[aria-selected='true'] {"
            " color:#0F172A !important; background:#E2E8F0 !important; border-color:#94A3B8 !important; }"
            "#insight-tabs [role='tabpanel'] { background:#FFFFFF !important; color:#1E293B !important; }"
            ".gradio-container .message {"
            " background:#F8FAFC !important; color:#1E293B !important; border:1px solid #E2E8F0 !important; }"
            ".gradio-container .message.user { background:#E0F2FE !important; }"
            ".gradio-container .message.bot, .gradio-container .message.assistant { background:#F8FAFC !important; }"
            "@media (max-width: 768px) {"
            "  #app-header h1 { font-size: 20px !important; line-height: 1.25; }"
            "  #app-header p { font-size: 13px !important; line-height: 1.55; }"
            "  #query-row { flex-direction: column !important; align-items: stretch !important; }"
            "  #query-input, #ask-button { width: 100% !important; min-width: 100% !important; }"
            "  #chatbot-panel { height: 360px !important; }"
            "  #graph-panel { min-height: 260px !important; }"
            "  #insight-tabs [role='tablist'] { display: grid !important;"
            " grid-template-columns: 1fr 1fr !important; gap: 6px; }"
            "  #insight-tabs [role='tablist'] button { min-width: 0 !important;"
            " width: 100% !important; padding: 8px 6px !important;"
            " font-size: 12px !important; }"
            "}"
        ),
    ) as demo:
        # ── Header ────────────────────────────────────────────────────────
        gr.HTML(
            "<div id='app-header' style='padding:14px 0 10px;border-bottom:1px solid #E2E8F0;margin-bottom:12px'>"
            "<h1 style='margin:0;font-size:24px;font-weight:700;color:#1E293B'>"
            "Riskfolio-Lib &mdash; GraphRAG + Agentic AI Demo</h1>"
            "<p style='margin:8px 0 6px;font-size:14px;color:#334155;line-height:1.7'>"
            "A production-style knowledge graph RAG system built over the"
            " <a href='https://riskfolio-lib.readthedocs.io/' target='_blank'"
            " style='color:#3B82F6;text-decoration:none'>Riskfolio-Lib</a>"
            " portfolio optimization library."
            " Entities (functions, classes, parameters, concepts) are extracted from"
            " source code and documentation and stored in Neo4j."
            " Each query runs a LangGraph agentic workflow"
            " &mdash; plan, retrieve, reason, verify &mdash; combining"
            " vector similarity search with"
            " graph-neighbourhood traversal for hybrid retrieval."
            "<br/><span style='font-size:12px;color:#475569'>"
            "Source code: "
            "<a href='https://github.com/ethantenison/riskfolio-graphrag-agent' target='_blank'"
            " style='color:#3B82F6;text-decoration:none'>"
            "github.com/ethantenison/riskfolio-graphrag-agent</a></span>"
            "</p>"
            "</div>"
        )

        # ── Example questions ───────────────────────────────────────────────────
        # question_box created with render=False so gr.Examples can
        # reference it before it's placed in the input row below.
        question_box = gr.Textbox(
            placeholder="Ask anything — e.g. 'What is a Risk Parity Portfolio?'",
            lines=2,
            scale=5,
            show_label=False,
            container=False,
            elem_id="query-input",
            render=False,
        )
        gr.Examples(
            examples=[
                ["What is portfolio optimization and why does it matter?"],
                ["How do I reduce risk in an investment portfolio?"],
                ["What constraints can I add to an optimization problem?"],
            ],
            inputs=[question_box],
            label="📋 Example questions — click any to load it below, then press Ask",
        )

        # ── Input bar ─────────────────────────────────────────────────────────────
        with gr.Row(equal_height=True, elem_id="query-row"):
            question_box.render()
            ask_button = gr.Button(
                "Ask  ↵",
                variant="primary",
                scale=0,
                min_width=100,
                elem_id="ask-button",
            )

        # ── Chat history ────────────────────────────────────────────────────────────
        chatbot = gr.Chatbot(
            label="",
            height=520,
            type="messages",
            show_label=False,
            bubble_full_width=False,
            elem_id="chatbot-panel",
        )
        gr.HTML(
            "<p style='margin:4px 0 8px;font-size:12px;color:#94A3B8'>"
            "\u26a0\ufe0f Demo only &mdash; not financial advice. "
            "Source library: "
            "<a href='https://riskfolio-lib.readthedocs.io/' target='_blank'"
            " style='color:#3B82F6'>Riskfolio-Lib docs</a>."
            "</p>"
        )

        # ── Under the Hood ───────────────────────────────────────────────────────────
        gr.Markdown(
            "### 🔍 Under the Hood: How the AI Works\n\n"
            "Each query goes through a multi-step agentic workflow. "
            "The tabs below show what happened when answering your question."
        )
        with gr.Tabs(elem_id="insight-tabs") as inner_tabs:
            with gr.Tab("Knowledge Graph"):
                gr.HTML(
                    "<p style='color:#64748B;font-size:12px;padding:4px 0 8px'>"
                    "Concepts and relationships retrieved from the Riskfolio-Lib"
                    " knowledge base for your query."
                    "</p>"
                )
                graph_panel = gr.Plot(
                    value=_render_graph_plot({"nodes": [], "edges": []}),
                    show_label=False,
                    elem_id="graph-panel",
                )
                graph_evidence_panel = gr.HTML(value=_EMPTY_GRAPH_EVIDENCE_HTML)

            with gr.Tab("Query Routing"):
                gr.HTML(
                    "<p style='color:#64748B;font-size:12px;padding:4px 0 8px'>"
                    "The agent breaks your question into sub-questions and"
                    " routes each to the best tool."
                    " <em>Dense</em> = vector search"
                    " &nbsp;|&nbsp; <em>Graph</em> = KG traversal"
                    " &nbsp;|&nbsp; <em>Hybrid</em> = both + reranking."
                    "</p>"
                )
                routing_panel = gr.HTML(value=_EMPTY_ROUTING_HTML)

            with gr.Tab("Answer Grounding"):
                gr.HTML(
                    "<p style='color:#64748B;font-size:12px;padding:4px 0 8px'>"
                    "Grounding checks that the answer is supported by retrieved"
                    " documents — not made up."
                    " Higher scores = stronger evidence."
                    "</p>"
                )
                grounding_panel = gr.HTML(value=_EMPTY_GROUNDING_HTML)
                citations_json = gr.JSON(label="Raw citation records", value=[])

            with gr.Tab("Governance"):
                gr.HTML(
                    "<p style='color:#64748B;font-size:12px;padding:4px 0 8px'>"
                    "LLM used, safety guardrails"
                    " (NL→Cypher injection prevention),"
                    " adaptive routing, and estimated cost per query."
                    "</p>"
                )
                governance_panel = gr.HTML(value=_EMPTY_GOVERNANCE_HTML)

            with gr.Tab("Benchmarking"):
                gr.HTML(
                    "<p style='color:#64748B;font-size:12px;padding:4px 0 8px'>"
                    "Contrastive baseline-vs-candidate evaluation summaries for interviews,"
                    " with a retrieval ablation fallback when the latest contrastive artifact"
                    " is not available."
                    "</p>"
                )
                contrastive_panel = gr.HTML(value=_EMPTY_CONTRASTIVE_HTML)

        _outputs = [
            question_box,
            chatbot,
            routing_panel,
            grounding_panel,
            citations_json,
            graph_evidence_panel,
            graph_panel,
            governance_panel,
            contrastive_panel,
            inner_tabs,
        ]
        _inputs = [question_box, chatbot]
        question_box.submit(_handle_submit, inputs=_inputs, outputs=_outputs)
        ask_button.click(_handle_submit, inputs=_inputs, outputs=_outputs)

    return demo


def launch_gradio_app(
    host: str = "127.0.0.1",
    port: int = 7860,
    top_k_default: int = 15,
    graph_max_nodes: int = 40,
    graph_max_edges: int = 80,
) -> None:
    app = create_gradio_app(
        top_k_default=top_k_default,
        graph_max_nodes=graph_max_nodes,
        graph_max_edges=graph_max_edges,
    )
    app.launch(server_name=host, server_port=port, show_api=False, share=True, ssr_mode=False)
