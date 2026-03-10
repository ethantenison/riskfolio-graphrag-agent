"""Document and source-code loader for the ingestion pipeline.

This loader performs structure-aware chunking for Riskfolio-Lib sources and
builds metadata-rich :class:`Document` chunks for graph extraction/retrieval.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# File extensions that are considered source material.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".py", ".rst", ".md", ".txt", ".ipynb"})


@dataclass
class Document:
    """A single text chunk produced by the loader.

    Attributes:
        content: The raw text of this chunk.
        source_path: Absolute path of the originating file.
        chunk_index: Zero-based index of this chunk within the source file.
        chunk_id: Canonical stable ID ``{relative_path}::chunk:{chunk_index}``.
        content_hash: Deterministic hash of normalized chunk content and source metadata.
        section: Human-readable section/symbol label for provenance.
        line_start: 1-based start line for this chunk.
        line_end: 1-based end line for this chunk.
        metadata: Arbitrary key/value pairs (e.g. module name, line range).
    """

    content: str
    source_path: str
    chunk_index: int = 0
    chunk_id: str = ""
    content_hash: str = ""
    section: str = ""
    line_start: int = 1
    line_end: int = 1
    metadata: dict[str, str | int] = field(default_factory=dict)


def load_directory(source_dir: str | Path, chunk_size: int = 1000, overlap: int = 100) -> list[Document]:
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
    source_root = source_path.resolve()
    for file_path in sorted(source_path.rglob("*")):
        if file_path.suffix not in SUPPORTED_EXTENSIONS or not file_path.is_file():
            continue
        file_docs = _chunk_file(
            file_path=file_path,
            source_root=source_root,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        documents.extend(file_docs)
        logger.debug("Loaded %d chunks from %s", len(file_docs), file_path)

    logger.info("Loaded %d total chunks from %s", len(documents), source_path)
    return documents


def _chunk_file(
    file_path: Path,
    source_root: Path,
    chunk_size: int,
    overlap: int,
) -> list[Document]:
    """Read a single file and split its text into overlapping chunks.

    Args:
        file_path: Path to the file to read.
        chunk_size: Maximum characters per chunk.
        overlap: Overlap in characters between consecutive chunks.

    Returns:
        A list of :class:`Document` instances.
    """
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Could not read %s: %s", file_path, exc)
        return []

    rel_path = file_path.resolve().relative_to(source_root).as_posix()
    source_type = _classify_source_type(file_path)
    module_name = _module_name(rel_path, source_type)

    base_metadata: dict[str, str | int] = {
        "source_type": source_type,
        "relative_path": rel_path,
        "filename": file_path.name,
        "extension": file_path.suffix,
        "module_name": module_name,
    }

    if source_type == "python" and file_path.suffix == ".py":
        return _chunk_python(text, file_path, base_metadata, chunk_size, overlap)
    if source_type == "test" and file_path.suffix == ".py":
        return _chunk_tests(text, file_path, base_metadata, chunk_size, overlap)
    if source_type == "example" and file_path.suffix == ".ipynb":
        return _chunk_example_notebook(text, file_path, base_metadata, chunk_size, overlap)
    if file_path.suffix in {".rst", ".md", ".txt"}:
        kind = "example_section" if source_type == "example" else "section"
        return _chunk_sections(text, file_path, base_metadata, kind, chunk_size, overlap)

    return _chunk_fallback(text, file_path, base_metadata, "fallback", chunk_size, overlap)


def _classify_source_type(file_path: Path) -> str:
    parts = set(file_path.parts)
    if "tests" in parts:
        return "test"
    if "examples" in parts:
        return "example"
    if "docs" in parts and "source" in parts:
        return "docs"
    return "python"


def _module_name(relative_path: str, source_type: str) -> str:
    base = relative_path
    if source_type == "python" and relative_path.startswith("riskfolio/src/"):
        base = relative_path.replace("riskfolio/src/", "", 1)
    stem = base.removesuffix(".py").replace("/", ".")
    if stem.endswith(".__init__"):
        stem = stem[: -len(".__init__")]
    return stem


def _chunk_python(
    text: str,
    file_path: Path,
    base_metadata: dict[str, str | int],
    chunk_size: int,
    overlap: int,
) -> list[Document]:
    lines = text.splitlines()
    chunks: list[Document] = []
    chunk_index = 0
    used_ranges: list[tuple[int, int]] = []

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _chunk_fallback(
            text,
            file_path,
            base_metadata,
            "fallback",
            chunk_size,
            overlap,
        )

    module_doc = ast.get_docstring(tree)
    if module_doc and tree.body and isinstance(tree.body[0], ast.Expr):
        expr = tree.body[0]
        if hasattr(expr, "lineno") and hasattr(expr, "end_lineno"):
            start = int(expr.lineno)
            end = int(expr.end_lineno or expr.lineno)
            used_ranges.append((start, end))
            chunk_index = _emit_line_chunk(
                chunks,
                lines,
                file_path,
                chunk_index,
                base_metadata,
                "module_docstring",
                start,
                end,
                chunk_size,
                overlap,
            )

    for node in ast.walk(tree):
        if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            continue
        if isinstance(node, ast.ClassDef):
            start = int(node.lineno)
            end = int(node.end_lineno or node.lineno)
            used_ranges.append((start, end))
            chunk_index = _emit_line_chunk(
                chunks,
                lines,
                file_path,
                chunk_index,
                {**base_metadata, "symbol_name": node.name},
                "class",
                start,
                end,
                chunk_size,
                overlap,
            )
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            start = int(node.lineno)
            end = int(node.end_lineno or node.lineno)
            used_ranges.append((start, end))
            chunk_index = _emit_line_chunk(
                chunks,
                lines,
                file_path,
                chunk_index,
                {**base_metadata, "symbol_name": node.name},
                "function",
                start,
                end,
                chunk_size,
                overlap,
            )

    chunk_index = _emit_uncovered_fallback(
        chunks,
        lines,
        file_path,
        chunk_index,
        base_metadata,
        used_ranges,
        chunk_size,
        overlap,
    )

    return chunks


def _chunk_tests(
    text: str,
    file_path: Path,
    base_metadata: dict[str, str | int],
    chunk_size: int,
    overlap: int,
) -> list[Document]:
    lines = text.splitlines()
    chunks: list[Document] = []
    chunk_index = 0

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _chunk_fallback(text, file_path, base_metadata, "fallback", chunk_size, overlap)

    test_nodes: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith("test_"):
            test_nodes.append(node)

    for node in sorted(test_nodes, key=lambda item: item.lineno):
        start = int(node.lineno)
        end = int(node.end_lineno or node.lineno)
        chunk_index = _emit_line_chunk(
            chunks,
            lines,
            file_path,
            chunk_index,
            {**base_metadata, "symbol_name": node.name},
            "test_function",
            start,
            end,
            chunk_size,
            overlap,
        )

    if not chunks:
        return _chunk_fallback(text, file_path, base_metadata, "fallback", chunk_size, overlap)
    return chunks


def _chunk_example_notebook(
    text: str,
    file_path: Path,
    base_metadata: dict[str, str | int],
    chunk_size: int,
    overlap: int,
) -> list[Document]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return _chunk_fallback(text, file_path, base_metadata, "example_section", chunk_size, overlap)

    cells = payload.get("cells", [])
    chunks: list[Document] = []
    chunk_index = 0
    for cell_index, cell in enumerate(cells, start=1):
        source = cell.get("source", [])
        if isinstance(source, list):
            content = "".join(source)
        else:
            content = str(source)
        if not content.strip():
            continue

        chunk_kind = "example_cell"
        if cell.get("cell_type") == "markdown":
            chunk_kind = "example_section"

        chunk_index = _emit_text_chunk(
            chunks,
            content,
            file_path,
            chunk_index,
            {
                **base_metadata,
                "chunk_kind": chunk_kind,
                "line_start": cell_index,
                "line_end": cell_index,
            },
            chunk_size,
            overlap,
        )

    return chunks


def _chunk_sections(
    text: str,
    file_path: Path,
    base_metadata: dict[str, str | int],
    default_chunk_kind: str,
    chunk_size: int,
    overlap: int,
) -> list[Document]:
    lines = text.splitlines()
    heading_lines = _detect_heading_lines(lines, file_path.suffix)
    if not heading_lines:
        return _chunk_fallback(
            text,
            file_path,
            base_metadata,
            default_chunk_kind,
            chunk_size,
            overlap,
        )

    heading_lines = sorted(set(heading_lines))
    if heading_lines and heading_lines[0] > 1:
        first_line = lines[0].strip() if lines else ""
        if first_line and not _is_heading_adornment_line(first_line):
            heading_lines = [1, *heading_lines]
    ranges: list[tuple[int, int]] = []
    for index, start in enumerate(heading_lines):
        end = heading_lines[index + 1] - 1 if index + 1 < len(heading_lines) else len(lines)
        if end >= start:
            ranges.append((start, end))

    chunks: list[Document] = []
    chunk_index = 0
    for start, end in ranges:
        section = _resolve_section_name(lines, start, end, default="section")
        chunk_index = _emit_line_chunk(
            chunks,
            lines,
            file_path,
            chunk_index,
            {**base_metadata, "section": section},
            default_chunk_kind,
            start,
            end,
            chunk_size,
            overlap,
        )

    return chunks


def _detect_heading_lines(lines: list[str], extension: str) -> list[int]:
    results: list[int] = []
    if extension == ".md":
        for index, line in enumerate(lines, start=1):
            if re.match(r"^#{1,6}\s+", line):
                results.append(index)
        return results

    for index in range(1, len(lines)):
        prev = lines[index - 1].strip()
        current = lines[index].strip()
        if not prev or not current:
            continue
        if _is_heading_adornment_line(current):
            results.append(index)
    return results


def _is_heading_adornment_line(line: str) -> bool:
    return bool(re.fullmatch(r"[=\-~^`:#\*]{3,}", line))


def _resolve_section_name(
    lines: list[str],
    line_start: int,
    line_end: int,
    default: str,
) -> str:
    for line in lines[line_start - 1 : line_end]:
        candidate = line.strip()
        if not candidate:
            continue
        if _is_heading_adornment_line(candidate):
            continue
        if candidate.startswith("#"):
            candidate = re.sub(r"^#{1,6}\s+", "", candidate).strip()
        if candidate:
            return candidate[:160]
    return default


def _chunk_fallback(
    text: str,
    file_path: Path,
    base_metadata: dict[str, str | int],
    chunk_kind: str,
    chunk_size: int,
    overlap: int,
) -> list[Document]:
    chunks: list[Document] = []
    _emit_text_chunk(
        chunks,
        text,
        file_path,
        0,
        {
            **base_metadata,
            "chunk_kind": chunk_kind,
            "line_start": 1,
            "line_end": max(1, text.count("\n") + 1),
        },
        chunk_size,
        overlap,
    )
    return chunks


def _emit_uncovered_fallback(
    chunks: list[Document],
    lines: list[str],
    file_path: Path,
    chunk_index: int,
    base_metadata: dict[str, str | int],
    used_ranges: list[tuple[int, int]],
    chunk_size: int,
    overlap: int,
) -> int:
    covered = [False] * (len(lines) + 1)
    for start, end in used_ranges:
        for line_num in range(max(1, start), min(len(lines), end) + 1):
            covered[line_num] = True

    start_line: int | None = None
    for line_num in range(1, len(lines) + 1):
        if not covered[line_num] and start_line is None:
            start_line = line_num
        if covered[line_num] and start_line is not None:
            chunk_index = _emit_line_chunk(
                chunks,
                lines,
                file_path,
                chunk_index,
                base_metadata,
                "fallback",
                start_line,
                line_num - 1,
                chunk_size,
                overlap,
            )
            start_line = None

    if start_line is not None:
        chunk_index = _emit_line_chunk(
            chunks,
            lines,
            file_path,
            chunk_index,
            base_metadata,
            "fallback",
            start_line,
            len(lines),
            chunk_size,
            overlap,
        )
    return chunk_index


def _emit_line_chunk(
    chunks: list[Document],
    lines: list[str],
    file_path: Path,
    chunk_index: int,
    base_metadata: dict[str, str | int],
    chunk_kind: str,
    line_start: int,
    line_end: int,
    chunk_size: int,
    overlap: int,
) -> int:
    if line_end < line_start:
        return chunk_index
    text = "\n".join(lines[line_start - 1 : line_end]).strip()
    if not text:
        return chunk_index
    return _emit_text_chunk(
        chunks,
        text,
        file_path,
        chunk_index,
        {
            **base_metadata,
            "chunk_kind": chunk_kind,
            "line_start": line_start,
            "line_end": line_end,
        },
        chunk_size,
        overlap,
    )


def _emit_text_chunk(
    chunks: list[Document],
    text: str,
    file_path: Path,
    chunk_index: int,
    metadata: dict[str, str | int],
    chunk_size: int,
    overlap: int,
) -> int:
    start = 0
    text_len = len(text)
    step = max(1, chunk_size - max(0, overlap))

    line_start = int(metadata.get("line_start", 1))
    line_end = int(metadata.get("line_end", line_start))
    total_lines = max(1, line_end - line_start + 1)
    relative_path = str(metadata.get("relative_path", file_path.name))
    section = str(metadata.get("section") or metadata.get("symbol_name") or metadata.get("chunk_kind") or "section")

    while start < text_len:
        end = min(text_len, start + chunk_size)
        chunk_text = text[start:end].strip()
        if chunk_text:
            start_ratio = start / max(1, text_len)
            end_ratio = end / max(1, text_len)
            sub_line_start = line_start + int(total_lines * start_ratio)
            sub_line_end = max(sub_line_start, line_start + int(total_lines * end_ratio))

            chunk_metadata: dict[str, str | int] = {
                **metadata,
                "line_start": sub_line_start,
                "line_end": min(line_end, sub_line_end),
                "section": section,
                "source_path": str(file_path),
            }
            resolved_line_end = int(chunk_metadata["line_end"])
            chunk_id = _build_chunk_id(relative_path=relative_path, chunk_index=chunk_index)
            content_hash = _build_content_hash(
                chunk_text=chunk_text,
                relative_path=relative_path,
                section=section,
                line_start=sub_line_start,
                line_end=resolved_line_end,
            )
            chunks.append(
                Document(
                    content=chunk_text,
                    source_path=str(file_path),
                    chunk_index=chunk_index,
                    chunk_id=chunk_id,
                    content_hash=content_hash,
                    section=section,
                    line_start=sub_line_start,
                    line_end=resolved_line_end,
                    metadata=chunk_metadata,
                )
            )
            chunk_index += 1

        if end >= text_len:
            break
        start += step

    return chunk_index


def _build_chunk_id(relative_path: str, chunk_index: int) -> str:
    return f"{relative_path}::chunk:{chunk_index}"


def _normalize_chunk_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def _build_content_hash(
    chunk_text: str,
    relative_path: str,
    section: str,
    line_start: int,
    line_end: int,
) -> str:
    payload = "|".join(
        (
            relative_path.strip(),
            section.strip(),
            str(line_start),
            str(line_end),
            _normalize_chunk_text(chunk_text),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def summarize_documents(documents: list[Document]) -> dict[str, Any]:
    """Return summary stats for loaded chunks.

    Returns file count, chunk count, and chunk/source-type breakdown.
    """
    files = {doc.source_path for doc in documents}
    by_source_type: dict[str, int] = {}
    for doc in documents:
        source_type = str(doc.metadata.get("source_type", "unknown"))
        by_source_type[source_type] = by_source_type.get(source_type, 0) + 1
    return {
        "files": len(files),
        "chunks": len(documents),
        "by_source_type": by_source_type,
    }
