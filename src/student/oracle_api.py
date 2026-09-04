from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from typing import List
import random
import json
import os

class Card(BaseModel):
    id: str
    url: str
    thumbnail: str
    content_type: str = "image/jpeg"

app = FastAPI(title="RAG Oracle Cards")

# Catalog path: repo-root/data/image_catalog.json
CATALOG_PATH = Path(__file__).resolve().parents[2] / "data" / "image_catalog.json"

def load_catalog() -> List[dict]:
    if not CATALOG_PATH.exists():
        return []
    with CATALOG_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)

@app.get("/api/oracle/cards", response_model=List[Card])
def get_cards(count: int = 12):
    """
    Return up to `count` random lightweight images from the catalog.
    Front-end can call this endpoint and render the returned URL/thumbnail.
    """
    catalog = load_catalog()
    if not catalog:
        raise HTTPException(status_code=404, detail="Image catalog not found")
    n = min(count, len(catalog))
    return random.sample(catalog, k=n)

@app.get("/api/oracle/cards/{card_id}", response_model=Card)
def get_card(card_id: str):
    catalog = load_catalog()
    for c in catalog:
        if c["id"] == card_id:
            return c
    raise HTTPException(status_code=404, detail="Card not found")
