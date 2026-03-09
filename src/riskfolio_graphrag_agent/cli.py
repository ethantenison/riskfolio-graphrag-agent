"""CLI entrypoint for riskfolio-graphrag-agent.

Commands
--------
ingest      Load and chunk source documents / code.
build-graph Build the knowledge graph in Neo4j.
graph-stats Show basic graph counts from Neo4j.
eval        Run retrieval-quality evaluation suite.
app         Start the interactive Gradio/FastAPI application.
"""

from pathlib import Path

import typer
import uvicorn
from rich.console import Console

from riskfolio_graphrag_agent.config.settings import Settings
from riskfolio_graphrag_agent.eval.evaluator import Evaluator, build_default_eval_samples
from riskfolio_graphrag_agent.eval.regression_gate import run_regression_gate
from riskfolio_graphrag_agent.graph.builder import GraphBuilder
from riskfolio_graphrag_agent.ingestion.loader import Document, load_directory, summarize_documents
from riskfolio_graphrag_agent.retrieval.retriever import HybridRetriever

app = typer.Typer(
    name="riskfolio-agent",
    help="Explainable GraphRAG agent over Riskfolio-Lib.",
    no_args_is_help=True,
)
console = Console()


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
) -> None:
    """Build (or rebuild) the knowledge graph in Neo4j.

    Args:
        drop_existing: When True, wipes the current graph before rebuilding.
    """
    settings = Settings()
    source_dirs = _resolve_focus_directories(source_dir, settings)
    documents = _load_from_directories(source_dirs)
    summary = summarize_documents(documents)

    builder = GraphBuilder(
        neo4j_uri=settings.neo4j_uri,
        neo4j_user=settings.neo4j_user,
        neo4j_password=settings.neo4j_password,
    )
    try:
        builder.build(documents, drop_existing=drop_existing)
    finally:
        builder.close()

    console.print(
        "[bold green]build-graph complete[/]",
        (
            f"files={summary['files']} chunks={summary['chunks']} "
            f"from {', '.join(str(path) for path in source_dirs)}"
        ),
    )
    by_source_type = summary.get("by_source_type", {})
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
        evaluator = Evaluator(samples=samples, retriever=retriever)
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
            f"relevance={report.answer_relevance:.3f}"
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

    console.print(f"[bold cyan]serve[/] starting API at http://{host}:{port}")
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    app()
