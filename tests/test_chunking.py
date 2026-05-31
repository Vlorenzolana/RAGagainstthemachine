"""Smoke tests for the chunking module."""

from __future__ import annotations

from student.chunking import chunk_python_code, chunk_text


def test_chunk_python_basic() -> None:
    src = (
        "import os\n\n"
        "def foo():\n    return 1\n\n"
        "class Bar:\n    def baz(self):\n        return 2\n"
    )
    spans = chunk_python_code(src, max_chunk_size=2000)
    assert spans
    for s, e in spans:
        assert 0 <= s < e <= len(src)


def test_chunk_python_oversized() -> None:
    src = "def f():\n" + ("    x = 1\n" * 500)
    spans = chunk_python_code(src, max_chunk_size=200)
    assert len(spans) > 1
    for s, e in spans:
        assert e - s <= 200 + 50  # tolerance for newline backoff


def test_chunk_text_paragraphs() -> None:
    src = "Para one.\n\nPara two is longer.\n\nPara three.\n"
    spans = chunk_text(src, max_chunk_size=20)
    joined = "".join(src[s:e] for s, e in spans)
    assert "Para one." in joined
    assert "Para three." in joined


def test_chunk_text_empty() -> None:
    assert chunk_text("", max_chunk_size=100) == []
    assert chunk_python_code("", max_chunk_size=100) == []
