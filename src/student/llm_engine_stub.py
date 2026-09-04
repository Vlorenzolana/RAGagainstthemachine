from __future__ import annotations

import hashlib

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Oracle LLM Engine Stub")


class GenerateRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    count: int = Field(default=12, ge=1, le=12)


class GenerateResponse(BaseModel):
    card_ids: list[str]
    explanation: str


def _deterministic_ids(question: str, count: int) -> list[str]:
    card_ids = [f"card-{index:02d}" for index in range(1, 13)]
    seed_bytes = hashlib.sha256(question.strip().lower().encode("utf-8")).digest()
    ranked_ids = sorted(
        card_ids,
        key=lambda card_id: hashlib.sha256(seed_bytes + card_id.encode("utf-8")).hexdigest(),
    )
    return ranked_ids[:count]


@app.post("/generate", response_model=GenerateResponse)
def generate(payload: GenerateRequest) -> GenerateResponse:
    return GenerateResponse(
        card_ids=_deterministic_ids(payload.question, payload.count),
        explanation="Deterministic stub selection based on question hash.",
    )
