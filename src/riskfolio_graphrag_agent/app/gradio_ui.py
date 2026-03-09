"""Gradio chat interface with Neo4j-backed graph visualisation."""

from __future__ import annotations

import html
import math
from typing import Any

from riskfolio_graphrag_agent.agent.workflow import AgentWorkflow
from riskfolio_graphrag_agent.config.settings import Settings
from riskfolio_graphrag_agent.graph.builder import GraphBuilder
from riskfolio_graphrag_agent.retrieval.retriever import HybridRetriever


def run_query_with_graph(
    question: str,
    top_k: int = 5,
    graph_max_nodes: int = 40,
    graph_max_edges: int = 80,
) -> tuple[str, list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    normalized_question = question.strip()
    if not normalized_question:
        return "Please enter a question.", [], {"nodes": [], "edges": []}

    settings = Settings()

    retriever = HybridRetriever(
        neo4j_uri=settings.neo4j_uri,
        neo4j_user=settings.neo4j_user,
        neo4j_password=settings.neo4j_password,
        top_k=max(1, int(top_k)),
        vector_store_backend=settings.vector_store_backend,
        chroma_persist_dir=settings.chroma_persist_dir,
        embedding_dim=settings.embedding_dim,
    )

    llm_generate = None
    if settings.openai_enable_generation and settings.openai_api_key.strip():
        from riskfolio_graphrag_agent.app.server import _make_openai_llm_generate

        llm_generate = _make_openai_llm_generate(settings)

    workflow = AgentWorkflow(
        retriever=retriever,
        model_name=settings.openai_model,
        llm_generate=llm_generate,
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
    return answer, state.citations, graph


def _render_graph_svg(graph: dict[str, list[dict[str, Any]]], size: int = 700) -> str:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    if not nodes:
        return "<div style='padding:12px;font-family:sans-serif;'>No graph data available for this query.</div>"

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
        edge_lines.append(
            f"<text x='{label_x:.1f}' y='{label_y:.1f}' font-size='10' fill='#5A6570'>"
            f"{edge_type}</text>"
        )

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

    return (
        f"<svg viewBox='0 0 {width} {height}' width='100%' height='100%' xmlns='http://www.w3.org/2000/svg'>"
        "<defs><marker id='arrow' markerWidth='10' markerHeight='7' refX='8' refY='3.5' orient='auto'>"
        "<polygon points='0 0, 10 3.5, 0 7' fill='#8AA0B8'/></marker></defs>"
        "<rect width='100%' height='100%' fill='white'/>"
        + "".join(edge_lines)
        + "".join(node_shapes)
        + "</svg>"
    )


def create_gradio_app(
    top_k_default: int = 5,
    graph_max_nodes: int = 40,
    graph_max_edges: int = 80,
):
    import gradio as gr

    def _handle_submit(
        question: str,
        history: list[tuple[str, str]] | None,
        top_k: int,
    ):
        answer, citations, graph = run_query_with_graph(
            question,
            top_k=top_k,
            graph_max_nodes=graph_max_nodes,
            graph_max_edges=graph_max_edges,
        )
        updated_history = list(history or [])
        if question.strip():
            updated_history.append((question, answer))
        graph_html = _render_graph_svg(graph)
        return "", updated_history, citations, graph_html

    with gr.Blocks(title="riskfolio-graphrag-agent") as demo:
        gr.Markdown("# Riskfolio GraphRAG Chat")
        gr.Markdown("Ask portfolio questions and inspect the connected Neo4j subgraph.")

        with gr.Row():
            chatbot = gr.Chatbot(label="Chat", height=520)
            with gr.Column():
                graph_panel = gr.HTML(label="Graph Visualisation", value=_render_graph_svg({"nodes": [], "edges": []}))
                citations_json = gr.JSON(label="Citations", value=[])

        with gr.Row():
            question_box = gr.Textbox(
                label="Question",
                placeholder="What is Hierarchical Risk Parity (HRP) in Riskfolio?",
                lines=2,
                scale=4,
            )
            top_k_slider = gr.Slider(
                minimum=1,
                maximum=20,
                value=max(1, int(top_k_default)),
                step=1,
                label="Top-k contexts",
                scale=1,
            )
            ask_button = gr.Button("Ask", variant="primary")

        question_box.submit(
            _handle_submit,
            inputs=[question_box, chatbot, top_k_slider],
            outputs=[question_box, chatbot, citations_json, graph_panel],
        )
        ask_button.click(
            _handle_submit,
            inputs=[question_box, chatbot, top_k_slider],
            outputs=[question_box, chatbot, citations_json, graph_panel],
        )

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
