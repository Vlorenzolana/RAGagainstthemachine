

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Sequence, Tuple

import bm25s
import numpy as np
from tqdm import tqdm

from student.models import Chunk, MinimalSource

DEFAULT_PROCESSED_DIR = Path("data/processed")
CHUNKS_FILE = "chunks/chunks.jsonl"
INDEX_DIR = "bm25_index"

# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "this", "that",
    "these", "those", "it", "its", "as", "by", "at", "from", "but", "if",
    "then", "so", "not", "no", "do", "does", "did", "how", "what", "which",
    "who", "whom", "why", "when", "where", "can", "could", "should", "would",
    "will", "shall", "may", "might", "must", "have", "has", "had",
}


def tokenize(text: str) -> List[str]:
    """Tokenise text with light code-awareness.

    Splits identifiers on underscores and CamelCase, lowercases, drops
    stopwords, keeps the original compound token too so exact matches still
    score highly.
    """
    out: List[str] = []
    for match in _TOKEN_RE.finditer(text):
        tok = match.group(0)
        lowered = tok.lower()
        if "_" in tok:
            parts = [p for p in tok.split("_") if p]
            out.append(lowered)
            for p in parts:
                pl = p.lower()
                if pl and pl not in _STOPWORDS:
                    out.append(pl)
            continue
        camel_parts = _CAMEL_RE.split(tok)
        if len(camel_parts) > 1:
            out.append(lowered)
            for p in camel_parts:
                pl = p.lower()
                if pl and pl not in _STOPWORDS:
                    out.append(pl)
        else:
            if lowered not in _STOPWORDS:
                out.append(lowered)
    return out


def tokenize_corpus(texts: Sequence[str]) -> List[List[str]]:
    """Tokenise a list of documents."""
    return [tokenize(t) for t in texts]


# ---------------------------------------------------------------------------
# Index persistence
# ---------------------------------------------------------------------------


def _chunks_path(processed_dir: Path) -> Path:
    return processed_dir / CHUNKS_FILE


def _index_dir(processed_dir: Path) -> Path:
    return processed_dir / INDEX_DIR


def save_chunks(chunks: Sequence[Chunk], processed_dir: Path) -> None:
    """Persist chunks to JSONL."""
    out = _chunks_path(processed_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")


def load_chunks(processed_dir: Path) -> List[Chunk]:
    """Load chunks from JSONL."""
    path = _chunks_path(processed_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No chunks found at {path}. Did you run `index` first?"
        )
    chunks: List[Chunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(Chunk.model_validate_json(line))
    return chunks


def build_and_save_index(chunks: Sequence[Chunk], processed_dir: Path) -> None:
    """Tokenise chunks, build a BM25 index and persist it."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    corpus_tokens = [
        tokenize(c.text) for c in tqdm(chunks, desc="Tokenizing", unit="chunk")
    ]
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)
    save_chunks(chunks, processed_dir)
    retriever.save(str(_index_dir(processed_dir)))


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


class BM25Retriever:
    """Loads a persisted BM25 index and answers top-k queries."""

    def __init__(self, processed_dir: Path = DEFAULT_PROCESSED_DIR) -> None:
        self.processed_dir = processed_dir
        self.chunks: List[Chunk] = load_chunks(processed_dir)
        self.retriever = bm25s.BM25.load(str(_index_dir(processed_dir)))

    def search(self, query: str, k: int = 10) -> List[Tuple[Chunk, float]]:
        """Return up to ``k`` (chunk, score) pairs ordered best first."""
        if not query.strip() or not self.chunks:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        k_eff = min(k, len(self.chunks))
        indices, scores = self.retriever.retrieve([tokens], k=k_eff, show_progress=False)
        idx_row = np.asarray(indices[0]).tolist()
        score_row = np.asarray(scores[0]).tolist()
        return [(self.chunks[i], float(s)) for i, s in zip(idx_row, score_row)]

    def search_to_sources(self, query: str, k: int = 10) -> List[MinimalSource]:
        """Convenience wrapper that returns ``MinimalSource`` records."""
        results = self.search(query, k=k)
        return [
            MinimalSource(
                file_path=c.file_path,
                first_character_index=c.first_character_index,
                last_character_index=c.last_character_index,
            )
            for c, _ in results
        ]


def load_chunks_only(processed_dir: Path = DEFAULT_PROCESSED_DIR) -> List[Chunk]:
    """Helper used by the answer pipeline when the retriever is loaded elsewhere."""
    return load_chunks(processed_dir)
