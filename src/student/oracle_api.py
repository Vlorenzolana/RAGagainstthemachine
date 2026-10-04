from __future__ import annotations

import json
import os
import random
import threading
import time
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from student.generation import AnswerGenerator, ContextSnippet, OpenRouterClient, OpenRouterError
from student.indexing import BM25Retriever, DEFAULT_PROCESSED_DIR
from student.models import MinimalSource


class Card(BaseModel):
    id: str
    url: str
    thumbnail: str
    content_type: str = "image/jpeg"


class ModelOption(BaseModel):
    id: str
    name: str
    context_length: int | None = None
    prompt_price: str | None = None
    completion_price: str | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=10, ge=1, le=50)
    model: str | None = None
    max_context_chars: int = Field(default=8000, ge=500, le=20000)
    max_new_tokens: int = Field(default=384, ge=32, le=2048)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class AskResponse(BaseModel):
    question: str
    answer: str
    retrieved_sources: List[MinimalSource]
    k: int
    model: str


app = FastAPI(title="RAG Oracle")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Catalog path: repo-root/data/image_catalog.json
CATALOG_PATH = Path(__file__).resolve().parents[2] / "data" / "image_catalog.json"
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "web" / "oracle"
MODEL_CACHE_TTL_SECONDS = 300

_model_cache_lock = threading.Lock()
_model_cache_timestamp = 0.0
_model_cache: List[ModelOption] = []


if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


def load_catalog() -> List[dict]:
    if not CATALOG_PATH.exists():
        return []
    with CATALOG_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _default_model() -> str:
    return os.environ.get("OPENROUTER_DEFAULT_MODEL", "openai/gpt-4o-mini")


def _get_client() -> OpenRouterClient:
    try:
        return OpenRouterClient()
    except OpenRouterError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _load_models(force_refresh: bool = False) -> List[ModelOption]:
    global _model_cache_timestamp, _model_cache
    now = time.time()
    if not force_refresh and _model_cache and now - _model_cache_timestamp < MODEL_CACHE_TTL_SECONDS:
        return _model_cache

    with _model_cache_lock:
        now = time.time()
        if not force_refresh and _model_cache and now - _model_cache_timestamp < MODEL_CACHE_TTL_SECONDS:
            return _model_cache

        client = _get_client()
        try:
            models = client.list_models()
        except OpenRouterError as exc:
            raise HTTPException(status_code=502, detail=f"Failed to fetch OpenRouter models: {exc}") from exc

        options = [
            ModelOption(
                id=m.id,
                name=m.name,
                context_length=m.context_length,
                prompt_price=m.prompt_price,
                completion_price=m.completion_price,
            )
            for m in models
        ]
        _model_cache = options
        _model_cache_timestamp = now
        return options


def _validate_or_default_model(model: str | None) -> str:
    selected = model or _default_model()
    options = _load_models()
    allowed = {option.id for option in options}
    if selected in allowed:
        return selected
    if selected == _default_model():
        return selected
    raise HTTPException(status_code=400, detail="Unsupported model id")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models", response_model=List[ModelOption])
def get_models() -> List[ModelOption]:
    return _load_models()


@app.post("/api/oracle/ask", response_model=AskResponse)
def ask_oracle(payload: AskRequest) -> AskResponse:
    model_id = _validate_or_default_model(payload.model)

    try:
        retriever = BM25Retriever(DEFAULT_PROCESSED_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to load BM25 index: {exc}") from exc

    results = retriever.search(payload.question, k=payload.k)
    snippets = [
        ContextSnippet(
            file_path=chunk.file_path,
            first_character_index=chunk.first_character_index,
            last_character_index=chunk.last_character_index,
            text=chunk.text,
        )
        for chunk, _ in results
    ]
    sources = [
        MinimalSource(
            file_path=chunk.file_path,
            first_character_index=chunk.first_character_index,
            last_character_index=chunk.last_character_index,
        )
        for chunk, _ in results
    ]

    generator = AnswerGenerator(
        model_id=_default_model(),
        max_new_tokens=payload.max_new_tokens,
        max_context_chars=payload.max_context_chars,
        temperature=payload.temperature,
        client=_get_client(),
    )
    try:
        answer_text = generator.generate(
            payload.question,
            snippets,
            sources,
            model=model_id,
        )
    except OpenRouterError as exc:
        raise HTTPException(status_code=502, detail=f"Generation failed: {exc}") from exc

    return AskResponse(
        question=payload.question,
        answer=answer_text,
        retrieved_sources=sources,
        k=payload.k,
        model=model_id,
    )


@app.get("/api/oracle/cards", response_model=List[Card])
def get_cards(count: int = 12) -> List[Card]:
    """Return up to `count` random lightweight images from the catalog."""
    catalog = load_catalog()
    if not catalog:
        raise HTTPException(status_code=404, detail="Image catalog not found")
    n = min(count, len(catalog))
    return random.sample(catalog, k=n)


@app.get("/api/oracle/cards/{card_id}", response_model=Card)
def get_card(card_id: str) -> Card:
    catalog = load_catalog()
    for c in catalog:
        if c["id"] == card_id:
            return Card.model_validate(c)
    raise HTTPException(status_code=404, detail="Card not found")
