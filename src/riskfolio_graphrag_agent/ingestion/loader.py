"""Document and source-code loader for the ingestion pipeline.

Responsibilities
----------------
1. Walk a source directory and discover ``.py`` / ``.rst`` / ``.md`` files.
2. Read and chunk each file into overlapping text windows.
3. Return a list of :class:`Document` objects ready for embedding and graph
   extraction.

This module currently provides **stub** implementations.  Replace the
``# TODO`` sections with real chunking and metadata extraction logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# File extensions that are considered source material.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".py", ".rst", ".md", ".txt"})


@dataclass
class Document:
    """A single text chunk produced by the loader.

    Attributes:
        content: The raw text of this chunk.
        source_path: Absolute path of the originating file.
        chunk_index: Zero-based index of this chunk within the source file.
        metadata: Arbitrary key/value pairs (e.g. module name, line range).
    """

    content: str
    source_path: str
    chunk_index: int = 0
    metadata: dict[str, str] = field(default_factory=dict)


def load_directory(
    source_dir: str | Path, chunk_size: int = 1000, overlap: int = 100
) -> list[Document]:
    """Walk *source_dir* and return a flat list of chunked :class:`Document` objects.

    Args:
        source_dir: Root directory to scan for supported files.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters to repeat between consecutive chunks.

    Returns:
        A list of :class:`Document` instances, one per chunk.

    Raises:
        FileNotFoundError: If *source_dir* does not exist.
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_path}")

    documents: list[Document] = []
    for file_path in sorted(source_path.rglob("*")):
        if file_path.suffix not in SUPPORTED_EXTENSIONS or not file_path.is_file():
            continue
        file_docs = _chunk_file(file_path, chunk_size=chunk_size, overlap=overlap)
        documents.extend(file_docs)
        logger.debug("Loaded %d chunks from %s", len(file_docs), file_path)

    logger.info("Loaded %d total chunks from %s", len(documents), source_path)
    return documents


def _chunk_file(file_path: Path, chunk_size: int, overlap: int) -> list[Document]:
    """Read a single file and split its text into overlapping chunks.

    Args:
        file_path: Path to the file to read.
        chunk_size: Maximum characters per chunk.
        overlap: Overlap in characters between consecutive chunks.

    Returns:
        A list of :class:`Document` instances.
    """
    # TODO: replace with a smarter splitter (e.g. sentence-aware, AST-based for .py)
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Could not read %s: %s", file_path, exc)
        return []

    chunks: list[Document] = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        chunks.append(
            Document(
                content=chunk_text,
                source_path=str(file_path),
                chunk_index=idx,
                metadata={"extension": file_path.suffix, "filename": file_path.name},
            )
        )
        start += chunk_size - overlap
        idx += 1

    return chunks
