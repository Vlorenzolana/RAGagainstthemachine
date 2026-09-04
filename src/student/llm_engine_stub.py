from __future__ import annotations

import random
from typing import Dict
from fastapi import FastAPI
from pydantic import BaseModel

class GenReq(BaseModel):
    question: str
    count: int = 12

class GenResp(BaseModel):
    cards: list
    explanation: str

app = FastAPI(title="LLM Engine Stub")

@app.post("/generate", response_model=GenResp)
def generate(req: GenReq):
    # Deterministic pseudo-selection for demo; replace with real model call.
    seed = sum(ord(c) for c in (req.question or ""))
    rng = random.Random(seed)
    # return simple ids "card-01".."card-12"
    all_ids = [f"card-{i:02d}" for i in range(1, 13)]
    chosen = rng.sample(all_ids, k=min(req.count or 12, len(all_ids)))
    return GenResp(cards=chosen, explanation="Motor stub: selección determinista basada en la pregunta")
