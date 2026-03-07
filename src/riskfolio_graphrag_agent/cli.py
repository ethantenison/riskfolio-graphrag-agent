"""CLI entrypoint for riskfolio-graphrag-agent.

Commands
--------
ingest      Load and chunk source documents / code.
build-graph Build the knowledge graph in Neo4j.
eval        Run retrieval-quality evaluation suite.
app         Start the interactive Gradio/FastAPI application.
"""

import typer
from rich.console import Console

app = typer.Typer(
    name="riskfolio-agent",
    help="Explainable GraphRAG agent over Riskfolio-Lib.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def ingest(
    source_dir: str = typer.Option(
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
    # TODO: wire up riskfolio_graphrag_agent.ingestion.loader
    console.print("[bold cyan]ingest[/] – placeholder. source_dir=", source_dir)


@app.command(name="build-graph")
def build_graph(
    drop_existing: bool = typer.Option(
        False, "--drop-existing", help="Drop all graph nodes before rebuilding."
    ),
) -> None:
    """Build (or rebuild) the knowledge graph in Neo4j.

    Args:
        drop_existing: When True, wipes the current graph before rebuilding.
    """
    # TODO: wire up riskfolio_graphrag_agent.graph.builder
    console.print("[bold cyan]build-graph[/] – placeholder. drop_existing=", drop_existing)


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
    # TODO: wire up riskfolio_graphrag_agent.eval.evaluator
    console.print("[bold cyan]eval[/] – placeholder. output_file=", output_file)


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
    # TODO: wire up riskfolio_graphrag_agent.app.server
    console.print(f"[bold cyan]serve[/] – placeholder. http://{host}:{port}")


if __name__ == "__main__":
    app()
