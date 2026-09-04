from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import requests
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

app = FastAPI(title="RAG Oracle Cards")

CATALOG_PATH = Path(__file__).resolve().parents[2] / "data" / "image_catalog.json"
FRONTEND_PATH = Path(__file__).resolve().parents[2] / "web" / "oracle"

if FRONTEND_PATH.exists():
    app.mount("/frontend", StaticFiles(directory=FRONTEND_PATH, html=True), name="frontend")


class Card(BaseModel):
    id: str
    url: str
    thumbnail: str
    content_type: str = "image/jpeg"


class OracleQuery(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    count: int = Field(default=12, ge=1, le=12)


class OracleAnswer(BaseModel):
    cards: list[Card]
    explanation: str


def _load_catalog() -> list[Card]:
    if not CATALOG_PATH.exists():
        return []
    with CATALOG_PATH.open("r", encoding="utf-8") as catalog_file:
        raw_catalog: Any = json.load(catalog_file)
    cards: list[Card] = []
    for entry in raw_catalog:
        cards.append(Card.model_validate(entry))
    return cards


def _require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    expected_key = os.getenv("API_KEY")
    if expected_key and x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _heuristic_card_ids(question: str, count: int, card_ids: list[str]) -> list[str]:
    seed_bytes = hashlib.sha256(question.strip().lower().encode("utf-8")).digest()
    ranked_ids = sorted(
        card_ids,
        key=lambda card_id: hashlib.sha256(seed_bytes + card_id.encode("utf-8")).hexdigest(),
    )
    return ranked_ids[:count]


def _engine_pick(question: str, count: int) -> tuple[list[str], str] | None:
    engine_url = os.getenv("LLM_ENGINE_URL")
    if not engine_url:
        return None

    url = f"{engine_url.rstrip('/')}/generate"
    try:
        response = requests.post(url, json={"question": question, "count": count}, timeout=3)
        response.raise_for_status()
        body = response.json()
    except requests.RequestException:
        return None

    card_ids = body.get("card_ids")
    if not isinstance(card_ids, list):
        return None
    sanitized_ids = [str(card_id) for card_id in card_ids]
    explanation = body.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        explanation = "Engine-selected cards"
    return sanitized_ids, explanation


def _select_cards(question: str, count: int, card_ids: list[str]) -> tuple[list[str], str]:
    max_count = min(count, len(card_ids))
    engine_result = _engine_pick(question, max_count)

    if engine_result is None:
        return (
            _heuristic_card_ids(question, max_count, card_ids),
            "Deterministic fallback used (engine unavailable).",
        )

    proposed_ids, explanation = engine_result
    selected_ids: list[str] = []
    seen: set[str] = set()
    for card_id in proposed_ids:
        if card_id in card_ids and card_id not in seen:
            selected_ids.append(card_id)
            seen.add(card_id)
        if len(selected_ids) >= max_count:
            break

    if len(selected_ids) < max_count:
        fallback_ids = _heuristic_card_ids(question, max_count, card_ids)
        for fallback_id in fallback_ids:
            if fallback_id not in seen:
                selected_ids.append(fallback_id)
                seen.add(fallback_id)
            if len(selected_ids) >= max_count:
                break

    return selected_ids, explanation


@app.get("/api/oracle/cards", response_model=list[Card])
def get_cards(
    count: int = Query(default=12, ge=1, le=12),
    _: None = Depends(_require_api_key),
) -> list[Card]:
    cards = _load_catalog()
    if not cards:
        raise HTTPException(status_code=404, detail="Image catalog not found")
    safe_count = min(count, len(cards))
    return cards[:safe_count]


@app.get("/api/oracle/cards/{card_id}", response_model=Card)
def get_card(card_id: str, _: None = Depends(_require_api_key)) -> Card:
    cards = _load_catalog()
    for card in cards:
        if card.id == card_id:
            return card
    raise HTTPException(status_code=404, detail="Card not found")


@app.post("/api/oracle/query", response_model=OracleAnswer)
def query_oracle(query: OracleQuery, _: None = Depends(_require_api_key)) -> OracleAnswer:
    cards = _load_catalog()
    if not cards:
        raise HTTPException(status_code=404, detail="Image catalog not found")

    by_id = {card.id: card for card in cards}
    selected_ids, explanation = _select_cards(query.question, query.count, list(by_id.keys()))
    selected_cards = [by_id[card_id] for card_id in selected_ids if card_id in by_id]

    return OracleAnswer(cards=selected_cards, explanation=explanation)
