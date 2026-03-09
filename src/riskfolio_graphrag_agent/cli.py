"""CLI entrypoint for riskfolio-graphrag-agent.

Commands
--------
ingest      Load and chunk source documents / code.
build-graph Build the knowledge graph in Neo4j.
graph-stats Show basic graph counts from Neo4j.
eval        Run retrieval-quality evaluation suite.
serve       Start the FastAPI server.
gradio      Start the Gradio chat interface.
"""

import json
import logging
import ssl
import time
from pathlib import Path
from urllib import request
from urllib.error import HTTPError, URLError

try:
    import certifi
except Exception:  # pragma: no cover - optional dependency fallback
    certifi = None

import typer
import uvicorn
from rich.console import Console

from riskfolio_graphrag_agent.config.settings import Settings
from riskfolio_graphrag_agent.eval.evaluator import Evaluator, build_default_eval_samples
from riskfolio_graphrag_agent.eval.regression_gate import run_regression_gate
from riskfolio_graphrag_agent.graph.builder import GraphBuilder
from riskfolio_graphrag_agent.ingestion.loader import Document, load_directory, summarize_documents
from riskfolio_graphrag_agent.retrieval.retriever import HybridRetriever
from riskfolio_graphrag_agent.runtime_ssl import initialize_ssl_truststore_once

app = typer.Typer(
    name="riskfolio-agent",
    help="Explainable GraphRAG agent over Riskfolio-Lib.",
    no_args_is_help=True,
)
console = Console()
logger = logging.getLogger(__name__)


def _build_ssl_context() -> ssl.SSLContext | None:
    if initialize_ssl_truststore_once():
        return None
    if certifi is None:
        return None
    return ssl.create_default_context(cafile=certifi.where())


def _configure_logging(log_level: str) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )


def _make_openai_graph_extractor(settings: Settings):
    def _extract(*, content: str, source_type: str, model_name: str) -> dict[str, object]:
        prompt = (
            "Extract Riskfolio graph entities and relationships from a source chunk. "
            "Return strict JSON with keys 'nodes' and 'edges'. "
            "Each node must include: label, name, properties. "
            "Each edge must include: source_name, source_label, target_name, target_label, relation_type. "
            "Use only known labels and relationships from the project ontology. "
            "Do not include explanations or markdown.\n\n"
            f"source_type: {source_type}\n"
            "known_node_labels: Chunk, DocPage, ExampleNotebook, TestCase, PythonModule, "
            "PythonClass, PythonFunction, Parameter, PortfolioMethod, RiskMeasure, ConstraintType, "
            "Estimator, ReportType, PlotType, Solver, Concept\n"
            "known_relationship_types: HAS_CHUNK, MENTIONS, DESCRIBES, DEMONSTRATES, VALIDATES, "
            "IMPLEMENTS, DECLARES, HAS_PARAMETER, USES, SUPPORTS_RISK_MEASURE, SUPPORTS_CONSTRAINT, "
            "USES_ESTIMATOR, PRODUCES_REPORT, RELATED_TO\n\n"
            f"content:\n{content[:6000]}"
        )

        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an information extraction engine. Output valid JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        endpoint = f"{settings.openai_base_url.rstrip('/')}/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            url=endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
        )

        max_attempts = max(1, int(settings.openai_retry_attempts) + 1)
        backoff_seconds = max(0.0, float(settings.openai_retry_backoff_seconds))
        raw = ""

        for attempt in range(1, max_attempts + 1):
            try:
                with request.urlopen(
                    http_request,
                    timeout=settings.openai_timeout_seconds,
                    context=_build_ssl_context(),
                ) as response:
                    raw = response.read().decode("utf-8")
                break
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="ignore")
                if exc.code in {408, 429, 500, 502, 503, 504} and attempt < max_attempts:
                    sleep_seconds = backoff_seconds * attempt
                    logger.warning(
                        "Graph LLM transient HTTP %s on attempt %d/%d; retrying in %.1fs",
                        exc.code,
                        attempt,
                        max_attempts,
                        sleep_seconds,
                    )
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)
                    continue
                raise RuntimeError(f"Graph LLM HTTP error {exc.code}: {detail}") from exc
            except (URLError, TimeoutError) as exc:
                if attempt < max_attempts:
                    sleep_seconds = backoff_seconds * attempt
                    logger.warning(
                        "Graph LLM request failed on attempt %d/%d (%s); retrying in %.1fs",
                        attempt,
                        max_attempts,
                        exc,
                        sleep_seconds,
                    )
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)
                    continue
                raise RuntimeError(f"Graph LLM endpoint unreachable: {exc}") from exc

        try:
            response_payload = json.loads(raw)
            choices = response_payload.get("choices", [])
            first_choice = choices[0] if isinstance(choices, list) and choices else {}
            message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
            content_text = message.get("content", "") if isinstance(message, dict) else ""
            if not isinstance(content_text, str) or not content_text.strip():
                return {"nodes": [], "edges": []}
            extracted = json.loads(content_text)
            return extracted if isinstance(extracted, dict) else {"nodes": [], "edges": []}
        except json.JSONDecodeError as exc:
            raise RuntimeError("Graph LLM returned invalid JSON") from exc

    return _extract


def _resolve_source_directories(source_dir: str | None, settings: Settings) -> list[Path]:
    candidate = Path(source_dir or settings.riskfolio_source_dir).expanduser().resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Source directory not found: {candidate}")

    dirs: list[Path] = []
    if (candidate / "riskfolio" / "src").exists() and (candidate / "docs" / "source").exists():
        dirs = [candidate / "riskfolio" / "src", candidate / "docs" / "source"]
    elif candidate.name == "src" and candidate.parent.name == "riskfolio":
        sibling_docs = candidate.parent.parent / "docs" / "source"
        dirs = [candidate]
        if sibling_docs.exists():
            dirs.append(sibling_docs)
    elif candidate.name == "source" and candidate.parent.name == "docs":
        sibling_src = candidate.parent.parent / "riskfolio" / "src"
        dirs = [candidate]
        if sibling_src.exists():
            dirs.append(sibling_src)
    else:
        dirs = [candidate]

    unique_dirs: list[Path] = []
    seen: set[str] = set()
    for directory in dirs:
        normalized = str(directory)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_dirs.append(directory)
    return unique_dirs


def _resolve_focus_directories(source_dir: str | None, settings: Settings) -> list[Path]:
    """Resolve Riskfolio-Lib focus directories for graph ingestion.

    Focus paths:
    - riskfolio/src
    - docs/source
    - examples
    - tests
    """
    root = Path(source_dir or settings.riskfolio_source_dir).expanduser().resolve()
    candidates = [
        root / "riskfolio" / "src",
        root / "docs" / "source",
        root / "examples",
        root / "tests",
    ]

    focus = [path for path in candidates if path.exists()]
    if focus:
        return focus

    parent = root.parent
    fallback_candidates = [
        parent / "riskfolio" / "src",
        parent / "docs" / "source",
        parent / "examples",
        parent / "tests",
    ]
    fallback_focus = [path for path in fallback_candidates if path.exists()]
    if fallback_focus:
        return fallback_focus

    if not root.exists():
        raise FileNotFoundError(f"Source directory not found: {root}")
    return [root]


def _load_from_directories(source_dirs: list[Path]) -> list[Document]:
    documents: list[Document] = []
    for directory in source_dirs:
        docs = load_directory(directory)
        documents.extend(docs)
    return documents


def _select_documents_for_build(
    documents: list[Document],
    *,
    chunk_offset: int = 0,
    max_chunks: int | None = None,
) -> list[Document]:
    if chunk_offset < 0:
        raise ValueError("chunk_offset must be >= 0")
    if max_chunks is not None and max_chunks <= 0:
        raise ValueError("max_chunks must be > 0 when provided")

    if not documents:
        return []

    start = min(chunk_offset, len(documents))
    if max_chunks is None:
        return documents[start:]
    end = start + max_chunks
    return documents[start:end]


@app.command()
def ingest(
    source_dir: str | None = typer.Option(
        None,
        "--source-dir",
        "-s",
        help="Path to Riskfolio-Lib source / docs (overrides RISKFOLIO_SOURCE_DIR).",
    ),
) -> None:
    """Load, chunk, and embed source documents and code.

    Args:
        source_dir: Optional path override for the source directory.
    """
    settings = Settings()
    source_dirs = _resolve_focus_directories(source_dir, settings)
    documents = _load_from_directories(source_dirs)
    summary = summarize_documents(documents)

    retriever = HybridRetriever(
        neo4j_uri=settings.neo4j_uri,
        neo4j_user=settings.neo4j_user,
        neo4j_password=settings.neo4j_password,
        vector_store_backend=settings.vector_store_backend,
        chroma_persist_dir=settings.chroma_persist_dir,
        embedding_dim=settings.embedding_dim,
    )
    try:
        upserted = retriever.upsert_documents(documents)
    finally:
        retriever.close()

    console.print(
        "[bold cyan]ingest[/]",
        f"files={summary['files']} chunks={summary['chunks']}",
        f"vector_upserted={upserted}",
        "from",
        ", ".join(str(path) for path in source_dirs),
    )
    by_source_type = summary.get("by_source_type", {})
    if isinstance(by_source_type, dict):
        for source_type, count in sorted(by_source_type.items()):
            console.print(f"  - {source_type}: {count}")


@app.command(name="build-graph")
def build_graph(
    drop_existing: bool = typer.Option(
        False, "--drop-existing", help="Drop all graph nodes before rebuilding."
    ),
    source_dir: str | None = typer.Option(
        None,
        "--source-dir",
        "-s",
        help="Path to Riskfolio-Lib source/docs root or subdirectory.",
    ),
    chunk_offset: int = typer.Option(
        0,
        "--chunk-offset",
        min=0,
        help="Skip this many chunks before graph extraction starts.",
    ),
    max_chunks: int | None = typer.Option(
        None,
        "--max-chunks",
        min=1,
        help="Limit graph extraction to at most this many chunks.",
    ),
) -> None:
    """Build (or rebuild) the knowledge graph in Neo4j.

    Args:
        drop_existing: When True, wipes the current graph before rebuilding.
    """
    initialize_ssl_truststore_once()
    settings = Settings()
    _configure_logging(settings.log_level)
    source_dirs = _resolve_focus_directories(source_dir, settings)
    documents = _load_from_directories(source_dirs)
    selected_documents = _select_documents_for_build(
        documents,
        chunk_offset=chunk_offset,
        max_chunks=max_chunks,
    )
    summary = summarize_documents(documents)
    selected_summary = summarize_documents(selected_documents)

    if not selected_documents:
        console.print(
            "[yellow]build-graph skipped[/]",
            "No chunks selected.",
            f"total_chunks={summary['chunks']} chunk_offset={chunk_offset} max_chunks={max_chunks}",
        )
        return

    builder = GraphBuilder(
        neo4j_uri=settings.neo4j_uri,
        neo4j_user=settings.neo4j_user,
        neo4j_password=settings.neo4j_password,
        llm_extract=(
            _make_openai_graph_extractor(settings)
            if settings.openai_enable_graph_extraction and settings.openai_api_key.strip()
            else None
        ),
        llm_model_name=settings.openai_model,
    )
    try:
        builder.build(selected_documents, drop_existing=drop_existing)
    finally:
        builder.close()

    console.print(
        "[bold green]build-graph complete[/]",
        (
            f"files={selected_summary['files']} chunks={selected_summary['chunks']} "
            f"(selected from total_chunks={summary['chunks']} offset={chunk_offset}) "
            f"from {', '.join(str(path) for path in source_dirs)}"
        ),
    )
    by_source_type = selected_summary.get("by_source_type", {})
    if isinstance(by_source_type, dict):
        for source_type, count in sorted(by_source_type.items()):
            console.print(f"  - {source_type}: {count}")


@app.command(name="graph-stats")
def graph_stats() -> None:
    """Print basic graph statistics from Neo4j."""
    settings = Settings()
    builder = GraphBuilder(
        neo4j_uri=settings.neo4j_uri,
        neo4j_user=settings.neo4j_user,
        neo4j_password=settings.neo4j_password,
    )
    try:
        stats = builder.get_stats()
    finally:
        builder.close()

    console.print(
        "[bold cyan]graph-stats[/]",
        f"nodes={stats['nodes']} relationships={stats['relationships']}",
    )
    label_counts = stats.get("node_counts_by_label", {})
    if isinstance(label_counts, dict) and label_counts:
        for label, count in label_counts.items():
            console.print(f"  - {label}: {count}")

    relationship_counts = stats.get("relationship_counts_by_type", {})
    if isinstance(relationship_counts, dict) and relationship_counts:
        console.print("[bold cyan]relationship-types[/]")
        for relationship_type, count in relationship_counts.items():
            console.print(f"  - {relationship_type}: {count}")


@app.command()
def eval(
    output_file: str = typer.Option(
        "eval_results.json", "--output", "-o", help="Path to write evaluation results JSON."
    ),
    metric_profile: str = typer.Option(
        "ragas-style",
        "--metric-profile",
        help="Evaluation metric profile to run: ragas-style or heuristic.",
        case_sensitive=False,
    ),
) -> None:
    """Run the retrieval-quality evaluation suite.

    Args:
        output_file: Path where the JSON results will be written.
    """
    settings = Settings()
    samples = build_default_eval_samples()
    retriever = HybridRetriever(
        neo4j_uri=settings.neo4j_uri,
        neo4j_user=settings.neo4j_user,
        neo4j_password=settings.neo4j_password,
        top_k=5,
    )

    try:
        normalized_profile = metric_profile.strip().lower()
        if normalized_profile not in {"ragas-style", "heuristic"}:
            raise typer.BadParameter("--metric-profile must be one of: ragas-style, heuristic")

        evaluator = Evaluator(
            samples=samples,
            retriever=retriever,
            metric_profile=normalized_profile,
        )
        report = evaluator.run()
        evaluator.save(output_file)
    finally:
        retriever.close()

    console.print(
        "[bold cyan]eval[/]",
        (
            f"samples={report.num_samples} recall={report.context_recall:.3f} "
            f"precision={report.context_precision:.3f} "
            f"faithfulness={report.answer_faithfulness:.3f} "
            f"relevance={report.answer_relevance:.3f} "
            f"profile={report.metric_profile}"
        ),
    )
    console.print(f"saved report to {output_file}")


@app.command(name="eval-gate")
def eval_gate(
    report_file: str = typer.Option(
        "eval_results.json", "--report", help="Path to eval report JSON."
    ),
    min_faithfulness: float = typer.Option(0.35, "--min-faithfulness"),
    min_relevance: float = typer.Option(0.8, "--min-relevance"),
    min_context_recall: float = typer.Option(0.45, "--min-context-recall"),
) -> None:
    """Fail fast if evaluation metrics regress below minimum thresholds."""
    run_regression_gate(
        report_path=report_file,
        min_faithfulness=min_faithfulness,
        min_relevance=min_relevance,
        min_context_recall=min_context_recall,
    )
    console.print("[bold green]eval-gate passed[/]")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host."),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port."),
) -> None:
    """Start the interactive application server.

    Args:
        host: Host address to bind.
        port: TCP port to listen on.
    """
    from riskfolio_graphrag_agent.app.server import create_app

    initialize_ssl_truststore_once()
    console.print(f"[bold cyan]serve[/] starting API at http://{host}:{port}")
    uvicorn.run(create_app(), host=host, port=port)


@app.command(name="gradio")
def gradio_app(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host."),
    port: int = typer.Option(7860, "--port", "-p", help="Bind port."),
    top_k: int = typer.Option(5, "--top-k", min=1, max=20, help="Top-k contexts per query."),
    graph_max_nodes: int = typer.Option(
        40,
        "--graph-max-nodes",
        min=1,
        max=200,
        help="Maximum nodes in the rendered graph.",
    ),
    graph_max_edges: int = typer.Option(
        80,
        "--graph-max-edges",
        min=1,
        max=400,
        help="Maximum edges in the rendered graph.",
    ),
) -> None:
    """Start the Gradio chat interface with graph visualisation."""
    from riskfolio_graphrag_agent.app.gradio_ui import launch_gradio_app

    initialize_ssl_truststore_once()
    console.print(f"[bold cyan]gradio[/] starting UI at http://{host}:{port}")
    launch_gradio_app(
        host=host,
        port=port,
        top_k_default=top_k,
        graph_max_nodes=graph_max_nodes,
        graph_max_edges=graph_max_edges,
    )


if __name__ == "__main__":
    app()
