

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple

from student.models import Chunk

# File extensions we consider useful for the vLLM repository.
CODE_EXTENSIONS = {".py"}
TEXT_EXTENSIONS = {".md", ".rst", ".txt"}
EXTRA_TEXT_EXTENSIONS = {".yaml", ".yml", ".toml", ".cfg", ".ini", ".json"}

# Directories we always skip when walking the corpus.
SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "node_modules",
    "build",
    "dist",
    ".tox",
    ".eggs",
}


def _is_binary(data: bytes) -> bool:
    """Heuristic: a file is binary if it contains NUL bytes in the first 1 KiB."""
    return b"\x00" in data[:1024]


def iter_corpus_files(root: Path) -> Iterator[Path]:
    """Yield every file under ``root`` that is worth indexing."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            path = Path(dirpath) / name
            ext = path.suffix.lower()
            if ext in CODE_EXTENSIONS or ext in TEXT_EXTENSIONS or ext in EXTRA_TEXT_EXTENSIONS:
                yield path


def read_file(path: Path) -> str:
    """Read a file as UTF-8, tolerating decoding errors and skipping binaries."""
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    if _is_binary(raw):
        return ""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Python code chunking
# ---------------------------------------------------------------------------


def _split_window(
    text: str, start: int, end: int, max_chunk_size: int
) -> List[Tuple[int, int]]:
    """Split a [start, end) slice into windows no larger than ``max_chunk_size``.

    Tries to break on newline boundaries to keep chunks readable.
    """
    spans: List[Tuple[int, int]] = []
    cur = start
    while cur < end:
        nxt = min(cur + max_chunk_size, end)
        if nxt < end:
            # Try to back off to the last newline within this window.
            nl = text.rfind("\n", cur + max_chunk_size // 2, nxt)
            if nl != -1 and nl > cur:
                nxt = nl + 1
        spans.append((cur, nxt))
        cur = nxt
    return spans


def chunk_python_code(text: str, max_chunk_size: int) -> List[Tuple[int, int]]:
    """Return chunk spans (start, end) for Python source ``text``.

    Splits on top-level ``def`` / ``class`` statements, then re-windows any
    individual chunk that exceeds ``max_chunk_size``. Falls back to plain
    windowing when the file cannot be parsed.
    """
    if not text:
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _split_window(text, 0, len(text), max_chunk_size)

    # Compute newline offsets so we can convert (line, col) -> char index.
    line_starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            line_starts.append(i + 1)
    line_starts.append(len(text) + 1)

    def offset(lineno: int, col: int) -> int:
        idx = max(0, min(lineno - 1, len(line_starts) - 1))
        return min(line_starts[idx] + col, len(text))

    spans: List[Tuple[int, int]] = []
    top_nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef,
                                                         ast.AsyncFunctionDef,
                                                         ast.ClassDef))]
    if not top_nodes:
        return _split_window(text, 0, len(text), max_chunk_size)

    # Header: everything before the first top-level def/class (imports, etc.).
    first_start = offset(top_nodes[0].lineno, top_nodes[0].col_offset)
    if first_start > 0:
        spans.extend(_split_window(text, 0, first_start, max_chunk_size))

    for i, node in enumerate(top_nodes):
        start = offset(node.lineno, node.col_offset)
        if i + 1 < len(top_nodes):
            end = offset(top_nodes[i + 1].lineno, top_nodes[i + 1].col_offset)
        else:
            end = len(text)
        if end - start <= max_chunk_size:
            spans.append((start, end))
        else:
            spans.extend(_split_window(text, start, end, max_chunk_size))

    # Drop empty spans and ensure ordering.
    return [(s, e) for s, e in spans if e > s]


# ---------------------------------------------------------------------------
# Text / Markdown chunking
# ---------------------------------------------------------------------------


_PARAGRAPH_RE = re.compile(r"\n\s*\n")


def chunk_text(text: str, max_chunk_size: int) -> List[Tuple[int, int]]:
    """Chunk Markdown / plain text on paragraph boundaries.

    Paragraphs are merged greedily so each chunk is as close to
    ``max_chunk_size`` characters as possible without exceeding it.
    Oversized paragraphs are sub-split with ``_split_window``.
    """
    if not text:
        return []

    # Build paragraph spans.
    paragraphs: List[Tuple[int, int]] = []
    cur = 0
    for match in _PARAGRAPH_RE.finditer(text):
        end = match.start()
        if end > cur:
            paragraphs.append((cur, end))
        cur = match.end()
    if cur < len(text):
        paragraphs.append((cur, len(text)))

    if not paragraphs:
        return _split_window(text, 0, len(text), max_chunk_size)

    spans: List[Tuple[int, int]] = []
    buf_start: int | None = None
    buf_end: int = 0
    for start, end in paragraphs:
        size = end - start
        if size > max_chunk_size:
            if buf_start is not None:
                spans.append((buf_start, buf_end))
                buf_start = None
            spans.extend(_split_window(text, start, end, max_chunk_size))
            continue
        if buf_start is None:
            buf_start, buf_end = start, end
            continue
        if end - buf_start <= max_chunk_size:
            buf_end = end
        else:
            spans.append((buf_start, buf_end))
            buf_start, buf_end = start, end
    if buf_start is not None:
        spans.append((buf_start, buf_end))

    return [(s, e) for s, e in spans if e > s]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_chunks(
    root: Path, max_chunk_size: int, files: Iterable[Path] | None = None
) -> List[Chunk]:
    """Walk ``root`` (or the supplied ``files``) and return all chunks.

    File paths in the resulting ``Chunk`` records are stored *relative* to
    ``root`` so they match the conventions used by the evaluation datasets.
    """
    chunks: List[Chunk] = []
    file_iter = files if files is not None else iter_corpus_files(root)
    cid = 0
    for path in file_iter:
        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = path
        rel_str = rel.as_posix()
        text = read_file(path)
        if not text:
            continue
        ext = path.suffix.lower()
        if ext in CODE_EXTENSIONS:
            spans = chunk_python_code(text, max_chunk_size)
            kind = "code"
        else:
            spans = chunk_text(text, max_chunk_size)
            kind = "text"
        for start, end in spans:
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    file_path=rel_str,
                    first_character_index=start,
                    last_character_index=end,
                    text=text[start:end],
                    kind=kind,
                )
            )
            cid += 1
    return chunks
