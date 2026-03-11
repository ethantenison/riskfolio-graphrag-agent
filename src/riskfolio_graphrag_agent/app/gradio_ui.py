"""Gradio chat interface with Neo4j-backed graph visualisation."""

from __future__ import annotations

import html
import math
from typing import Any

import gradio as gr

from riskfolio_graphrag_agent.agent.workflow import AgentWorkflow
from riskfolio_graphrag_agent.config.settings import Settings
from riskfolio_graphrag_agent.graph.builder import GraphBuilder
from riskfolio_graphrag_agent.retrieval.embeddings import resolve_embedding_provider
from riskfolio_graphrag_agent.retrieval.retriever import HybridRetriever
from riskfolio_graphrag_agent.retrieval.router import QueryToolRouter


def run_query_with_graph(
    question: str,
    top_k: int = 5,
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
        graph = graph_builder.get_query_subgraph(
            normalized_question,
            max_nodes=max(1, int(graph_max_nodes)),
            max_edges=max(1, int(graph_max_edges)),
        )
    except Exception:
        graph = {"nodes": [], "edges": []}
    finally:
        graph_builder.close()

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
        },
        "governance": {
            "model": settings.openai_model,
            "base_retrieval_mode": settings.retrieval_mode,
            "adaptive_routing_enabled": settings.adaptive_tool_routing_enabled,
            "vector_backend": settings.vector_store_backend,
            "sub_questions": list(state.sub_questions),
            "estimated_cost_usd": estimated_cost,
        },
    }


def _render_graph_svg(graph: dict[str, list[dict[str, Any]]], size: int = 480) -> str:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    if not nodes:
        return (
            "<div style='display:flex;align-items:center;justify-content:center;"
            f"height:{size}px;background:#F8FAFC;border-radius:8px;color:#9CA3AF;font-size:13px;font-family:sans-serif'>"
            "No graph data available &mdash; submit a query to populate the knowledge graph view."
            "</div>"
        )

    width = size
    height = size
    radius = max(220, int(size * 0.38))
    center_x = width / 2
    center_y = height / 2

    positions: dict[str, tuple[float, float]] = {}
    total = len(nodes)
    for index, node in enumerate(nodes):
        node_id = str(node.get("id", ""))
        angle = (2 * math.pi * index) / max(total, 1)
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        positions[node_id] = (x, y)

    edge_lines: list[str] = []
    for edge in edges:
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if source not in positions or target not in positions:
            continue
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        edge_type = html.escape(str(edge.get("type", "")))
        label_x = (x1 + x2) / 2
        label_y = (y1 + y2) / 2
        edge_lines.append(
            f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' "
            "stroke='#8AA0B8' stroke-width='1.8' stroke-opacity='0.7' marker-end='url(#arrow)'/>"
        )
        edge_lines.append(f"<text x='{label_x:.1f}' y='{label_y:.1f}' font-size='10' fill='#5A6570'>{edge_type}</text>")

    node_shapes: list[str] = []
    for node in nodes:
        node_id = str(node.get("id", ""))
        x, y = positions[node_id]
        name = str(node.get("name", "")) or "Unnamed"
        labels = node.get("labels", [])
        primary_label = ""
        if isinstance(labels, list) and labels:
            primary_label = str(labels[0])
        source_path = str(node.get("source_path", ""))
        title = html.escape(f"{name} [{primary_label}]\\n{source_path}")
        display_name = html.escape(name[:26] + ("…" if len(name) > 26 else ""))
        label_text = html.escape(primary_label)
        node_shapes.append(
            f"<g><title>{title}</title>"
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='20' fill='#3B82F6' fill-opacity='0.9'/>"
            f"<text x='{x:.1f}' y='{y + 36:.1f}' text-anchor='middle' font-size='10' fill='#1E293B'>{display_name}</text>"
            f"<text x='{x:.1f}' y='{y + 49:.1f}' text-anchor='middle' font-size='9' fill='#64748B'>{label_text}</text>"
            "</g>"
        )

    svg = (
        f"<svg viewBox='0 0 {width} {height}' width='100%' height='100%' xmlns='http://www.w3.org/2000/svg'>"
        "<defs><marker id='arrow' markerWidth='10' markerHeight='7' refX='8' refY='3.5' orient='auto'>"
        "<polygon points='0 0, 10 3.5, 0 7' fill='#8AA0B8'/></marker></defs>"
        "<rect width='100%' height='100%' fill='white'/>" + "".join(edge_lines) + "".join(node_shapes) + "</svg>"
    )
    return f"<div style='width:100%;height:{size}px;overflow:hidden;border-radius:8px;border:1px solid #E2E8F0'>{svg}</div>"


# ── Insight-panel rendering helpers ─────────────────────────────────────────

_MODE_COLOURS: dict[str, str] = {
    "dense": "#3B82F6",
    "sparse": "#8B5CF6",
    "graph": "#10B981",
    "hybrid_rerank": "#F59E0B",
}

_EMPTY_ROUTING_HTML = (
    "<p style='color:#9CA3AF;font-size:13px;padding:8px'>Submit a query to see adaptive tool-selection routing decisions.</p>"
)
_EMPTY_GROUNDING_HTML = (
    "<p style='color:#9CA3AF;font-size:13px;padding:8px'>Submit a query to see grounding and faithfulness metrics.</p>"
)
_EMPTY_GRAPH_EVIDENCE_HTML = (
    "<p style='color:#9CA3AF;font-size:13px;padding:8px'>Submit a query to see graph entities and neighbourhood evidence.</p>"
)
_EMPTY_GOVERNANCE_HTML = (
    "<p style='color:#9CA3AF;font-size:13px;padding:8px'>Submit a query to see governance metrics and cost estimates.</p>"
)


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


# ─────────────────────────────────────────────────────────────────────────────


def create_gradio_app(
    top_k_default: int = 5,
    graph_max_nodes: int = 40,
    graph_max_edges: int = 80,
):

    def _handle_submit(
        question: str,
        history: list[dict[str, str]] | None,
        top_k: int,
    ):
        answer, citations, graph, insights = run_query_with_graph(
            question,
            top_k=top_k,
            graph_max_nodes=graph_max_nodes,
            graph_max_edges=graph_max_edges,
        )
        updated_history = list(history or [])
        if question.strip():
            updated_history.append({"role": "user", "content": question})
            updated_history.append({"role": "assistant", "content": answer})
        return (
            "",
            updated_history,
            _render_routing_html(insights),
            _render_grounding_html(insights),
            citations,
            _render_graph_evidence_html(insights),
            _render_graph_svg(graph),
            _render_governance_html(insights),
        )

    with gr.Blocks(
        title="Riskfolio GraphRAG",
        theme=gr.themes.Soft(),
        css="footer {display:none !important}",
    ) as demo:
        # ── Header ────────────────────────────────────────────────────────
        gr.HTML(
            "<div style='padding:14px 0 6px'>"
            "<h1 style='margin:0;font-size:22px;font-weight:700;color:#1E293B'>"
            "Riskfolio GraphRAG</h1>"
            "<p style='margin:5px 0 0;font-size:13px;color:#64748B'>"
            "Agentic GraphRAG over Riskfolio-Lib &mdash; "
            "adaptive tool routing &middot; knowledge-graph retrieval &middot; "
            "grounded LLM answers &middot; governed, cited, explainable."
            "</p></div>"
        )

        # ── Primary: full-width chat ───────────────────────────────────────
        chatbot = gr.Chatbot(
            label="",
            height=460,
            type="messages",
            show_label=False,
            bubble_full_width=False,
        )

        # ── Input bar ─────────────────────────────────────────────────────
        with gr.Row(equal_height=True):
            question_box = gr.Textbox(
                placeholder="Ask a Riskfolio question — e.g. 'How does HRP compare to MVO?'",
                lines=2,
                scale=5,
                show_label=False,
                container=False,
            )
            with gr.Column(scale=1, min_width=160):
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=max(1, int(top_k_default)),
                    step=1,
                    label="Top-k results",
                )
                ask_button = gr.Button("Ask  \u21b5", variant="primary")

        # ── Interview Signals (Tabs) ───────────────────────────────────────
        with gr.Accordion("\U0001f50d  Interview Signals", open=True):
            gr.Markdown(
                "_Per-query signals: adaptive tool-selection routing, RAG faithfulness grounding, "
                "knowledge-graph entity evidence, and production governance controls — "
                "aligned with the Dell KG / RAG Agentic AI Expert role._"
            )
            with gr.Tabs():
                with gr.Tab("\U0001f9ed  Routing"):
                    routing_panel = gr.HTML(value=_EMPTY_ROUTING_HTML)

                with gr.Tab("\u2705  Grounding"):
                    grounding_panel = gr.HTML(value=_EMPTY_GROUNDING_HTML)
                    citations_json = gr.JSON(label="Raw citations", value=[])

                with gr.Tab("\U0001f578  Graph Evidence"):
                    graph_evidence_panel = gr.HTML(value=_EMPTY_GRAPH_EVIDENCE_HTML)
                    graph_panel = gr.HTML(value=_render_graph_svg({"nodes": [], "edges": []}))

                with gr.Tab("\U0001f6e1  Governance"):
                    governance_panel = gr.HTML(value=_EMPTY_GOVERNANCE_HTML)

        _outputs = [
            question_box,
            chatbot,
            routing_panel,
            grounding_panel,
            citations_json,
            graph_evidence_panel,
            graph_panel,
            governance_panel,
        ]
        question_box.submit(_handle_submit, inputs=[question_box, chatbot, top_k_slider], outputs=_outputs)
        ask_button.click(_handle_submit, inputs=[question_box, chatbot, top_k_slider], outputs=_outputs)

    return demo


def launch_gradio_app(
    host: str = "127.0.0.1",
    port: int = 7860,
    top_k_default: int = 5,
    graph_max_nodes: int = 40,
    graph_max_edges: int = 80,
) -> None:
    app = create_gradio_app(
        top_k_default=top_k_default,
        graph_max_nodes=graph_max_nodes,
        graph_max_edges=graph_max_edges,
    )
    app.launch(server_name=host, server_port=port, show_api=False)
