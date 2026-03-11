"""Gradio chat interface with Neo4j-backed graph visualisation."""

from __future__ import annotations

import html
import json
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


def _load_visjs() -> str:
    """Load vis-network JS from the bundled static file."""
    import pathlib

    static = pathlib.Path(__file__).parent / "static" / "vis-network.min.js"
    return static.read_text(encoding="utf-8")


def _render_graph_visjs(graph: dict[str, list[dict[str, Any]]], height: int = 520) -> str:
    """Render an interactive vis.js network: drag nodes, scroll to zoom, pan."""
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    if not nodes:
        return (
            "<div style='display:flex;align-items:center;justify-content:center;"
            f"height:{height}px;background:#F8FAFC;border-radius:8px;color:#9CA3AF;"
            "font-size:13px;font-family:sans-serif'>"
            "Ask a question to populate the knowledge graph."
            "</div>"
        )

    vis_nodes: list[dict] = []
    present_types: list[str] = []
    seen_types: set[str] = set()

    for node in nodes:
        nid = str(node.get("id", ""))
        name = str(node.get("name", "")).strip() or "Unnamed"
        labels = node.get("labels", [])
        primary = str(labels[0]) if labels else "Concept"
        colour = _NODE_COLOURS.get(primary, _DEFAULT_NODE_COLOUR)
        source = str(node.get("source_path", ""))
        tooltip_html = f"<b>{html.escape(name)}</b><br/>Type: {html.escape(primary)}" + (
            f"<br/><small style='color:#94A3B8'>{html.escape(source)}</small>" if source else ""
        )
        if primary not in seen_types:
            seen_types.add(primary)
            present_types.append(primary)
        vis_nodes.append(
            {
                "id": nid,
                "label": name[:26] + ("\u2026" if len(name) > 26 else ""),
                "_tooltip": tooltip_html,
                "color": {
                    "background": colour,
                    "border": "#ffffff",
                    "highlight": {"background": colour, "border": "#1E293B"},
                    "hover": {"background": colour, "border": "#1E293B"},
                },
                "font": {
                    "size": 11,
                    "color": "#1E293B",
                    "strokeWidth": 3,
                    "strokeColor": "#fff",
                },
                "size": 18,
            }
        )

    vis_edges: list[dict] = []
    for i, edge in enumerate(edges):
        src = str(edge.get("source", ""))
        tgt = str(edge.get("target", ""))
        if src == tgt:
            continue
        rel = str(edge.get("type", ""))
        colour = _REL_COLOURS.get(rel, _DEFAULT_REL_COLOUR)
        vis_edges.append(
            {
                "id": i,
                "from": src,
                "to": tgt,
                "label": rel,
                "color": {"color": colour, "highlight": colour, "hover": colour},
                "font": {
                    "size": 9,
                    "color": colour,
                    "strokeWidth": 2,
                    "strokeColor": "#fff",
                },
                "arrows": "to",
                "width": 1.5,
                "smooth": {"type": "curvedCW", "roundness": 0.15},
            }
        )

    nodes_json = json.dumps(vis_nodes)
    edges_json = json.dumps(vis_edges)

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_rows = "".join(
        "<div style='display:flex;align-items:center;gap:5px;margin:2px 0'>"
        f"<span style='width:10px;height:10px;border-radius:50%;flex-shrink:0;"
        f"background:{_NODE_COLOURS.get(pt, _DEFAULT_NODE_COLOUR)};display:inline-block'>"
        f"</span><span style='font-size:10px;color:#334155'>{html.escape(pt)}</span></div>"
        for pt in sorted(present_types)
    )
    legend_html = (
        "<div style='position:absolute;bottom:8px;left:8px;"
        "background:rgba(248,250,252,0.92);border:1px solid #E2E8F0;"
        "border-radius:6px;padding:8px 12px;max-height:180px;overflow-y:auto;z-index:10'>"
        "<div style='font-size:9px;font-weight:700;color:#64748B;"
        f"text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px'>Node Types</div>"
        f"{legend_rows}</div>"
    )

    inner_html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'/>"
        f"<script>{_load_visjs()}</script><style>"
        "body{margin:0;padding:0;background:#F8FAFC;overflow:hidden;position:relative}"
        f"#g{{width:100%;height:{height}px}}"
        "#tt{position:absolute;pointer-events:none;display:none;background:#1E293B;"
        "color:#F8FAFC;font-size:12px;padding:6px 10px;border-radius:6px;max-width:240px;"
        "z-index:999;box-shadow:0 2px 8px rgba(0,0,0,.3);line-height:1.5;"
        "font-family:system-ui,sans-serif}"
        f"</style></head><body><div id='g'></div>{legend_html}"
        "<div id='tt'></div><script>"
        f"var nodes=new vis.DataSet({nodes_json});"
        f"var edges=new vis.DataSet({edges_json});"
        "var opt={physics:{enabled:true,barnesHut:{gravitationalConstant:-7000,"
        "centralGravity:0.25,springLength:130,springConstant:0.04},"
        "stabilization:{iterations:200,fit:true}},"
        "interaction:{hover:true,zoomView:true,dragNodes:true,dragView:true},"
        "nodes:{shape:'dot',borderWidth:2,borderWidthSelected:3},"
        "edges:{selectionWidth:3},layout:{improvedLayout:true}};"
        "var tt=document.getElementById('tt');"
        "function initNet(){"
        "var c=document.getElementById('g');"
        "var net=new vis.Network(c,{nodes:nodes,edges:edges},opt);"
        "net.on('hoverNode',function(p){"
        "var n=nodes.get(p.node);"
        "if(n&&n._tooltip){tt.innerHTML=n._tooltip;tt.style.display='block';"
        "tt.style.left=(p.pointer.DOM.x+12)+'px';tt.style.top=(p.pointer.DOM.y-10)+'px'}"
        "});"
        "net.on('blurNode',function(){tt.style.display='none'});"
        "net.on('dragStart',function(){tt.style.display='none'});"
        "net.on('zoom',function(){tt.style.display='none'});"
        "net.once('stabilizationIterationsDone',function(){"
        "net.fit({animation:false});"
        "});"
        "}"
        "var c=document.getElementById('g');"
        "if(c.offsetWidth>0){initNet();}"
        "else{"
        "var ro=new ResizeObserver(function(){"
        "var c2=document.getElementById('g');"
        "if(c2.offsetWidth>0){ro.disconnect();initNet();}"
        "});"
        "ro.observe(c);"
        "}"
        "</script></body></html>"
    )

    srcdoc = html.escape(inner_html, quote=True)
    return (
        f'<iframe srcdoc="{srcdoc}" '
        f'style="width:100%;height:{height}px;border:none;border-radius:8px;'
        f'background:#F8FAFC" sandbox="allow-scripts"></iframe>'
    )


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
            parts.append(f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='#94A3B8' stroke-width='1.5'/>")
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
            _render_graph_visjs({"nodes": [], "edges": []}),
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
            _render_graph_visjs(graph),
            _render_governance_html(insights),
            gr.update(selected=0),
        )

    with gr.Blocks(
        title="Portfolio AI Assistant — Knowledge Graph + RAG",
        theme=gr.themes.Soft(),
        css=(
            "@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');"
            "* { font-family: 'Roboto', sans-serif !important; }"
            "footer {display:none !important}"
        ),
    ) as demo:
        # ── Header ────────────────────────────────────────────────────────
        gr.HTML(
            "<div style='padding:14px 0 10px;border-bottom:1px solid #E2E8F0;margin-bottom:12px'>"
            "<h1 style='margin:0;font-size:24px;font-weight:700;color:#1E293B'>"
            "Riskfolio-Lib &mdash; GraphRAG + Agentic AI Demo</h1>"
            "<p style='margin:8px 0 6px;font-size:14px;color:#334155;line-height:1.7'>"
            "A <strong>production-style knowledge graph RAG system</strong> built over the"
            " <a href='https://riskfolio-lib.readthedocs.io/' target='_blank'"
            " style='color:#3B82F6;text-decoration:none'>Riskfolio-Lib</a>"
            " portfolio optimization library."
            " Entities (functions, classes, parameters, concepts) are extracted from"
            " source code and documentation and stored in <strong>Neo4j</strong>."
            " Each query runs a <strong>LangGraph agentic workflow</strong>"
            " &mdash; plan, retrieve, reason, verify &mdash; combining"
            " <strong>vector similarity search</strong> with"
            " <strong>graph-neighbourhood traversal</strong> for hybrid retrieval."
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
        with gr.Row(equal_height=True):
            question_box.render()
            ask_button = gr.Button("Ask  ↵", variant="primary", scale=0, min_width=100)

        # ── Chat history ────────────────────────────────────────────────────────────
        chatbot = gr.Chatbot(
            label="",
            height=520,
            type="messages",
            show_label=False,
            bubble_full_width=False,
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
        with gr.Tabs() as inner_tabs:
            with gr.Tab("🕸  Knowledge Graph"):
                gr.HTML(
                    "<p style='color:#64748B;font-size:12px;padding:4px 0 8px'>"
                    "Concepts and relationships retrieved from the Riskfolio-Lib"
                    " knowledge base for your query."
                    "</p>"
                )
                graph_panel = gr.HTML(value=_render_graph_visjs({"nodes": [], "edges": []}))
                graph_evidence_panel = gr.HTML(value=_EMPTY_GRAPH_EVIDENCE_HTML)

            with gr.Tab("🧭  Query Routing"):
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

            with gr.Tab("✅  Answer Grounding"):
                gr.HTML(
                    "<p style='color:#64748B;font-size:12px;padding:4px 0 8px'>"
                    "Grounding checks that the answer is supported by retrieved"
                    " documents — not made up."
                    " Higher scores = stronger evidence."
                    "</p>"
                )
                grounding_panel = gr.HTML(value=_EMPTY_GROUNDING_HTML)
                citations_json = gr.JSON(label="Raw citation records", value=[])

            with gr.Tab("🛡  Governance"):
                gr.HTML(
                    "<p style='color:#64748B;font-size:12px;padding:4px 0 8px'>"
                    "LLM used, safety guardrails"
                    " (NL→Cypher injection prevention),"
                    " adaptive routing, and estimated cost per query."
                    "</p>"
                )
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
    app.launch(server_name=host, server_port=port, show_api=False, share=True)
