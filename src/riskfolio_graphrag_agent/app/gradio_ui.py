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


def _render_graph_svg(graph: dict[str, list[dict[str, Any]]], size: int = 680) -> str:
    from collections import defaultdict

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    if not nodes:
        return (
            "<div style='display:flex;align-items:center;justify-content:center;"
            f"height:{size}px;background:#F8FAFC;border-radius:8px;color:#9CA3AF;"
            "font-size:13px;font-family:sans-serif'>"
            "No graph data available &mdash; submit a query to populate the knowledge graph view."
            "</div>"
        )

    W, H = 920, 860
    CX, CY = W / 2, (H - 80) / 2  # leave 80 px at bottom for legend

    # ── Group nodes by primary label ─────────────────────────────────────────
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        labels = node.get("labels", [])
        primary = str(labels[0]) if labels else "Concept"
        groups[primary].append(node)

    group_keys = sorted(groups, key=lambda k: -len(groups[k]))
    num_groups = len(group_keys)

    # Arrange group centres in a circle; scale radius with group count
    outer_r = min(330, max(140, num_groups * 38))
    group_centers: dict[str, tuple[float, float]] = {}
    for i, gk in enumerate(group_keys):
        angle = 2 * math.pi * i / max(num_groups, 1) - math.pi / 2
        group_centers[gk] = (CX + outer_r * math.cos(angle), CY + outer_r * math.sin(angle))

    # Place individual nodes in a sub-circle within each group
    NODE_R = 16
    positions: dict[str, tuple[float, float]] = {}
    for gk, gnodes in groups.items():
        gx, gy = group_centers[gk]
        n = len(gnodes)
        sub_r = max(NODE_R + 4, min(55, (NODE_R + 6) * n / max(math.pi, 1)))
        for j, node in enumerate(gnodes):
            nid = str(node.get("id", ""))
            if n == 1:
                positions[nid] = (gx, gy)
            else:
                a = 2 * math.pi * j / n - math.pi / 2
                positions[nid] = (gx + sub_r * math.cos(a), gy + sub_r * math.sin(a))

    # ── Edges ────────────────────────────────────────────────────────────────
    seen_colours: set[str] = set()
    edge_svgs: list[str] = []
    for edge in edges:
        src = str(edge.get("source", ""))
        tgt = str(edge.get("target", ""))
        if src not in positions or tgt not in positions or src == tgt:
            continue
        x1, y1 = positions[src]
        x2, y2 = positions[tgt]
        rel = str(edge.get("type", ""))
        colour = _REL_COLOURS.get(rel, _DEFAULT_REL_COLOUR)
        seen_colours.add(colour)
        marker_id = f"arr_{colour.lstrip('#')}"

        # Perpendicular control point for a subtle curve
        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx * dx + dy * dy) or 1
        nx, ny = -dy / length, dx / length
        offset = min(22, max(8, length * 0.15))
        qx = (x1 + x2) / 2 + nx * offset
        qy = (y1 + y2) / 2 + ny * offset

        # Trim start/end so path begins/ends at circle perimeter
        d0x, d0y = qx - x1, qy - y1
        d0len = math.sqrt(d0x**2 + d0y**2) or 1
        sx = x1 + d0x / d0len * (NODE_R + 2)
        sy = y1 + d0y / d0len * (NODE_R + 2)
        d1x, d1y = x2 - qx, y2 - qy
        d1len = math.sqrt(d1x**2 + d1y**2) or 1
        ex = x2 - d1x / d1len * (NODE_R + 5)
        ey = y2 - d1y / d1len * (NODE_R + 5)

        # Label at t=0.5 on the bezier: 0.25*P0 + 0.5*Pctrl + 0.25*P2
        lx = 0.25 * x1 + 0.5 * qx + 0.25 * x2
        ly = 0.25 * y1 + 0.5 * qy + 0.25 * y2

        safe_rel = html.escape(rel)
        title = html.escape(f"{edge.get('source', '')} ─{rel}→ {edge.get('target', '')}")
        edge_svgs.append(
            f"<g><title>{title}</title>"
            f"<path d='M {sx:.1f} {sy:.1f} Q {qx:.1f} {qy:.1f} {ex:.1f} {ey:.1f}' "
            f"fill='none' stroke='{colour}' stroke-width='1.5' stroke-opacity='0.7' "
            f"marker-end='url(#{marker_id})'/>"
            f"<text x='{lx:.1f}' y='{ly:.1f}' text-anchor='middle' font-size='8.5' fill='{colour}' "
            f"paint-order='stroke' stroke='white' stroke-width='2.5' stroke-linejoin='round'>"
            f"{safe_rel}</text>"
            f"</g>"
        )

    # Arrow marker defs — one per unique edge colour
    markers = "".join(
        f"<marker id='arr_{c[1:]}' markerWidth='7' markerHeight='5' "
        f"refX='6' refY='2.5' orient='auto'>"
        f"<polygon points='0 0, 7 2.5, 0 5' fill='{c}'/></marker>"
        for c in seen_colours
    )

    # ── Nodes ────────────────────────────────────────────────────────────────
    node_svgs: list[str] = []
    for node in nodes:
        nid = str(node.get("id", ""))
        if nid not in positions:
            continue
        x, y = positions[nid]
        name = str(node.get("name", "")).strip() or "Unnamed"
        labels = node.get("labels", [])
        primary = str(labels[0]) if labels else "Concept"
        fill = _NODE_COLOURS.get(primary, _DEFAULT_NODE_COLOUR)
        source_path = str(node.get("source_path", ""))
        tooltip = html.escape(f"{name}\nType: {primary}" + (f"\n{source_path}" if source_path else ""))
        display = html.escape(name[:20] + ("…" if len(name) > 20 else ""))
        node_svgs.append(
            f"<g><title>{tooltip}</title>"
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{NODE_R}' fill='{fill}' "
            f"stroke='white' stroke-width='2' fill-opacity='0.93'/>"
            f"<text x='{x:.1f}' y='{y + NODE_R + 12:.1f}' text-anchor='middle' font-size='9' "
            f"font-weight='600' fill='#1E293B' "
            f"paint-order='stroke' stroke='white' stroke-width='2.5' stroke-linejoin='round'>"
            f"{display}</text>"
            f"</g>"
        )

    # ── Legend ───────────────────────────────────────────────────────────────
    present = sorted({str(n.get("labels", ["Concept"])[0]) if n.get("labels") else "Concept" for n in nodes})
    cols_per_row = 5
    leg_row_h = 18
    leg_top = H - 72
    leg_items: list[str] = []
    for idx, pt in enumerate(present):
        col = idx % cols_per_row
        row = idx // cols_per_row
        lx = 16 + col * 178
        ly = leg_top + row * leg_row_h + 10
        c = _NODE_COLOURS.get(pt, _DEFAULT_NODE_COLOUR)
        leg_items.append(
            f"<circle cx='{lx + 6}' cy='{ly}' r='5' fill='{c}'/>"
            f"<text x='{lx + 15}' y='{ly + 4}' font-size='10' fill='#334155'>{html.escape(pt)}</text>"
        )

    leg_rows = math.ceil(len(present) / cols_per_row)
    leg_bg_h = leg_rows * leg_row_h + 12
    legend = (
        f"<rect x='8' y='{leg_top - 4}' width='{W - 16}' height='{leg_bg_h}' "
        f"rx='5' fill='#F8FAFC' stroke='#E2E8F0' stroke-width='1'/>" + "".join(leg_items)
    )

    svg = (
        f"<svg viewBox='0 0 {W} {H}' width='100%' height='100%' "
        f"xmlns='http://www.w3.org/2000/svg' font-family='system-ui,sans-serif'>"
        f"<defs>{markers}</defs>"
        "<rect width='100%' height='100%' fill='white'/>" + "".join(edge_svgs) + "".join(node_svgs) + legend + "</svg>"
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


def _render_summary_card(insights: dict, citations: list) -> str:
    """One-line 'what just happened' strip shown below the chatbot."""
    grounding = insights.get("grounding", {})
    graph_ev = insights.get("graph_evidence", {})
    gov = insights.get("governance", {})
    n_sources = grounding.get("citation_count", 0)
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
        f"<span>\U0001f4da <strong>{n_sources}</strong> sources retrieved</span>"
        f"<span>\U0001f578 <strong>{subgraph_nodes}</strong> graph nodes</span>"
        f"<span>\U0001f4ca avg score <strong>{avg_score:.3f}</strong></span>"
        f"<span style='color:{v_colour}'><strong>{v_icon}</strong></span>"
        f"<span>\U0001f4b0 est. cost <strong>${cost:.6f}</strong></span>"
        "</div>"
    )


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
        """Generator: first yield shows loading state, second yield shows results."""
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
            _render_graph_svg({"nodes": [], "edges": []}),
            _LOADING_HTML,
            gr.update(open=False),
            "",
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
        has_graph = bool(graph.get("nodes"))
        tab_update = gr.update(selected=2) if has_graph else gr.update(selected=0)
        yield (
            "",
            final_history,
            _render_routing_html(insights),
            _render_grounding_html(insights),
            citations,
            _render_graph_evidence_html(insights),
            _render_graph_svg(graph),
            _render_governance_html(insights),
            gr.update(open=True),
            _render_summary_card(insights, citations),
            tab_update,
        )

    with gr.Blocks(
        title="Portfolio AI Assistant — Knowledge Graph + RAG",
        theme=gr.themes.Soft(),
        css="footer {display:none !important}",
    ) as demo:
        # ── Header ────────────────────────────────────────────────────────
        gr.HTML(
            "<div style='padding:14px 0 6px'>"
            "<h1 style='margin:0;font-size:22px;font-weight:700;color:#1E293B'>"
            "Portfolio AI Assistant</h1>"
            "<p style='margin:6px 0 0;font-size:14px;color:#334155;line-height:1.6'>"
            "Ask plain-English questions about"
            " <strong>investment portfolio construction</strong> &mdash;"
            " how to balance risk and return, which strategies to use,"
            " and how parameters interact."
            " Answers are grounded in a <strong>knowledge graph</strong>"
            " built from the Riskfolio-Lib library."
            "</p>"
            "<p style='margin:4px 0 0;font-size:12px;color:#94A3B8'>"
            "\u26a0\ufe0f Demo only &mdash; not financial advice. "
            "Source library: "
            "<a href='https://riskfolio-lib.readthedocs.io/' target='_blank'"
            " style='color:#3B82F6'>Riskfolio-Lib docs</a>."
            "</p></div>"
        )

        # ── What is portfolio optimization? (collapsible explainer) ───────
        with gr.Accordion("💡  New here? What is portfolio optimization?", open=False):
            gr.Markdown(
                """
**Portfolio optimization** decides *how to split money across investments*
to get the best return for a given level of risk (or least risk for a desired return).
Think of it like packing a suitcase: maximum value, within a weight limit.

**Key terms:**
| Term | Plain-English meaning |
|---|---|
| **MVO (Mean-Variance Optimization)** | Classic: find the "perfect" mix using returns and how assets co-move |
| **HRP (Hierarchical Risk Parity)** | Modern: groups similar assets, spreads risk evenly — robust to noisy data |
| **CVaR / Conditional Value at Risk** | "Worst-case loss" measure — how bad can things get in a bad market? |
| **Covariance / Correlation** | How much two assets move together — low correlation = better diversification |
| **Efficient Frontier** | All best possible portfolios — each trades off risk vs return |
| **Rebalancing** | Resetting to target weights as prices drift over time |

**This demo** uses an AI knowledge graph to answer questions — not just guessing.
                """
            )

        # ── Primary: full-width chat ───────────────────────────────────────
        chatbot = gr.Chatbot(
            label="",
            height=460,
            type="messages",
            show_label=False,
            bubble_full_width=False,
        )
        summary_card = gr.HTML(value="")

        # ── Input bar ─────────────────────────────────────────────────────
        with gr.Row(equal_height=True):
            question_box = gr.Textbox(
                placeholder="Ask anything — e.g. 'How does HRP compare to MVO?'",
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
                    label="Sources to retrieve",
                )
                ask_button = gr.Button("Ask  \u21b5", variant="primary")

        # ── Example questions (defined after question_box so it can be referenced) ──
        with gr.Accordion("📋  Example questions — click any to load", open=True):
            gr.Markdown("_Click a question to load it into the box above, then press **Ask**._")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Beginner — no prior knowledge needed**")
                    gr.Examples(
                        examples=[
                            ["What is portfolio optimization and why does it matter?"],
                            ["How do I reduce risk in an investment portfolio?"],
                            ["What is diversification and how does it work in practice?"],
                        ],
                        inputs=[question_box],
                        label=None,
                    )
                with gr.Column():
                    gr.Markdown("**Intermediate — comparing strategies**")
                    gr.Examples(
                        examples=[
                            ["How does HRP compare to classical MVO?"],
                            ["When should I use CVaR instead of variance?"],
                            ["What constraints can I add to an optimization problem?"],
                        ],
                        inputs=[question_box],
                        label=None,
                    )
                with gr.Column():
                    gr.Markdown("**Advanced — implementation details**")
                    gr.Examples(
                        examples=[
                            ["Which functions in Riskfolio-Lib implement HRP?"],
                            ["What parameters does the rp_optimization function accept?"],
                            ["What factor models are available for covariance estimation?"],
                        ],
                        inputs=[question_box],
                        label=None,
                    )

        # ── Under the Hood: AI Capabilities ───────────────────────────────
        under_hood_accordion = gr.Accordion("\U0001f50d  Under the Hood: How the AI Works", open=False)
        with under_hood_accordion:
            gr.Markdown(
                """
Each query goes through a multi-step agentic workflow.
The tabs below show what happened when answering your question:

- **Query Routing** — decides *which tool* to use per sub-question
  (vector search, graph traversal, or hybrid)
- **Answer Grounding** — verifies the answer is backed by evidence,
  not hallucinated
- **Knowledge Graph** — shows which entities and relationships were used
- **Governance & Cost** — model, guardrails, and token cost
                """
            )
            with gr.Tabs() as inner_tabs:
                with gr.Tab("\U0001f9ed  Query Routing"):
                    gr.HTML(
                        "<p style='color:#64748B;font-size:12px;padding:4px 0 8px'>"
                        "The agent breaks your question into sub-questions and "
                        "routes each to the best tool. "
                        "<em>Dense</em> = vector search"
                        " &nbsp;|&nbsp; <em>Graph</em> = KG traversal"
                        " &nbsp;|&nbsp; <em>Hybrid</em> = both + reranking."
                        "</p>"
                    )
                    routing_panel = gr.HTML(value=_EMPTY_ROUTING_HTML)

                with gr.Tab("\u2705  Answer Grounding"):
                    gr.HTML(
                        "<p style='color:#64748B;font-size:12px;padding:4px 0 8px'>"
                        "Grounding checks that the answer is supported by retrieved"
                        " documents — not made up. Higher scores = stronger evidence."
                        "</p>"
                    )
                    grounding_panel = gr.HTML(value=_EMPTY_GROUNDING_HTML)
                    citations_json = gr.JSON(label="Raw citation records", value=[])

                with gr.Tab("\U0001f578  Knowledge Graph"):
                    gr.HTML(
                        "<p style='color:#64748B;font-size:12px;padding:4px 0 8px'>"
                        "The knowledge graph stores concepts, functions, and parameters"
                        " extracted from Riskfolio-Lib source and docs."
                        " Nodes are colored by type; edges show relationships."
                        "</p>"
                    )
                    graph_evidence_panel = gr.HTML(value=_EMPTY_GRAPH_EVIDENCE_HTML)
                    graph_panel = gr.HTML(value=_render_graph_svg({"nodes": [], "edges": []}))

                with gr.Tab("\U0001f6e1  Governance"):
                    gr.HTML(
                        "<p style='color:#64748B;font-size:12px;padding:4px 0 8px'>"
                        "LLM used, safety guardrails (NL\u2192Cypher injection prevention),"
                        " adaptive routing status, and estimated cost per query."
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
            under_hood_accordion,
            summary_card,
            inner_tabs,
        ]
        _inputs = [question_box, chatbot, top_k_slider]
        question_box.submit(_handle_submit, inputs=_inputs, outputs=_outputs)
        ask_button.click(_handle_submit, inputs=_inputs, outputs=_outputs)

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
