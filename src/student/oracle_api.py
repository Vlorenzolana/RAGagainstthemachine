from __future__ import annotations

import os
import json
import random
from pathlib import Path
from typing import Any, List

import requests
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

# --- Models --------------------------------------------------------------
class Card(BaseModel):
    id: str
    url: str
    thumbnail: str
    content_type: str = "image/jpeg"

class QueryRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=500)
    count: int = Field(default=12, ge=1, le=12)

class QueryResponse(BaseModel):
    question: str
    cards: List[Card]
    explanation: str

# --- App & paths --------------------------------------------------------
app = FastAPI(title="RAG Oracle Cards")

CATALOG_PATH = Path(__file__).resolve().parents[2] / "data" / "image_catalog.json"
STATIC_DIR = Path(__file__).resolve().parents[2] / "web" / "oracle"
if STATIC_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(STATIC_DIR)), name="frontend")

_cors_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

# --- Simple API-key dependency (optional) -------------------------------
API_KEY = os.environ.get("API_KEY")  # if not set, API is open (dev convenience)

def require_api_key(x_api_key: str | None = Header(default=None)) -> bool:
    if API_KEY:
        if not x_api_key or x_api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True

# --- Helpers ------------------------------------------------------------
def load_catalog() -> List[dict[str, Any]]:
    if not CATALOG_PATH.exists():
        return []
    with CATALOG_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)

# --- Engine call --------------------------------------------------------
LLM_ENGINE_URL = os.environ.get("LLM_ENGINE_URL")  # e.g. http://engine:8001


def call_engine(question: str, count: int) -> dict[str, Any] | None:
    """Call external engine service. Expected to return JSON { cards: [...], explanation: '...' }"""
    if not LLM_ENGINE_URL:
        return None
    try:
        resp = requests.post(
            f"{LLM_ENGINE_URL.rstrip('/')}/generate",
            json={"question": question, "count": count},
            timeout=3.0,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        # Fail open to local heuristic fallback
        return None
    return None

# --- Endpoints ----------------------------------------------------------
@app.get("/api/oracle/cards", response_model=List[Card], dependencies=[Depends(require_api_key)])
def get_cards(count: int = Query(default=12, ge=1, le=12)) -> List[dict[str, Any]]:
    catalog = load_catalog()
    if not catalog:
        raise HTTPException(status_code=404, detail="Image catalog not found")
    n = min(count, len(catalog))
    return random.sample(catalog, k=n)

@app.get("/api/oracle/cards/{card_id}", response_model=Card, dependencies=[Depends(require_api_key)])
def get_card(card_id: str) -> dict[str, Any]:
    catalog = load_catalog()
    for c in catalog:
        if c["id"] == card_id:
            return c
    raise HTTPException(status_code=404, detail="Card not found")

@app.post("/api/oracle/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
def query_oracle(req: QueryRequest) -> QueryResponse:
    """
    Lightweight query endpoint. Tries the external engine service first (if configured),
    otherwise falls back to a deterministic, CPU-free selection heuristic.
    """
    catalog = load_catalog()
    if not catalog:
        raise HTTPException(status_code=404, detail="Image catalog not found")
    count = min(req.count, len(catalog))

    # Try engine
    engine_resp = call_engine(req.question, count)
    if engine_resp and isinstance(engine_resp, dict) and engine_resp.get("cards"):
        # engine is expected to return full card objects or at least IDs
        cards = engine_resp["cards"]
        explanation = engine_resp.get("explanation", "Seleccionadas por motor externo")
        # if engine returned ids, map to full card entries
        if cards and isinstance(cards[0], str):
            idset = set(cards)
            chosen = [c for c in catalog if c["id"] in idset]
            return QueryResponse(question=req.question, cards=chosen, explanation=explanation)
        return QueryResponse(question=req.question, cards=cards, explanation=explanation)

    # Fallback deterministic pseudo-random lightweight selection
    seed = sum(ord(c) for c in req.question)
    rng = random.Random(seed)
    chosen = rng.sample(catalog, k=count)
    explanation = (
        "Seleccionadas por heurística ligera local (simulación). "
        "Sustituir por motor IA cuando sea necesario."
    )
    return QueryResponse(question=req.question, cards=chosen, explanation=explanation)
