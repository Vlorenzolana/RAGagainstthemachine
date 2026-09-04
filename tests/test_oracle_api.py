from __future__ import annotations

import json
from pathlib import Path

import pytest

from student import oracle_api
from student import llm_engine_stub
from fastapi import HTTPException
from student.oracle_api import OracleQuery


@pytest.fixture
def temp_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, str]]:
    catalog = [
        {
            "id": f"card-{idx:02d}",
            "url": f"https://example.test/full/{idx}",
            "thumbnail": f"https://example.test/thumb/{idx}",
            "content_type": "image/jpeg",
        }
        for idx in range(1, 13)
    ]
    catalog_path = tmp_path / "image_catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    monkeypatch.setattr(oracle_api, "CATALOG_PATH", catalog_path)
    return catalog


def test_get_cards_returns_requested_count(temp_catalog: list[dict[str, str]]) -> None:
    cards = oracle_api.get_cards(count=5)
    assert len(cards) == 5
    assert cards[0].id == temp_catalog[0]["id"]


def test_query_fallback_is_deterministic(
    temp_catalog: list[dict[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LLM_ENGINE_URL", raising=False)
    first = oracle_api.query_oracle(OracleQuery(question="favourite song", count=4))
    second = oracle_api.query_oracle(OracleQuery(question="favourite song", count=4))

    first_ids = [card.id for card in first.cards]
    second_ids = [card.id for card in second.cards]

    assert first_ids == second_ids
    assert "fallback" in first.explanation.lower()


def test_query_uses_engine_response_when_available(
    temp_catalog: list[dict[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "card_ids": ["card-02", "card-04", "unknown"],
                "explanation": "Engine-selected cards",
            }

    def fake_post(*args: object, **kwargs: object) -> FakeResponse:
        return FakeResponse()

    monkeypatch.setenv("LLM_ENGINE_URL", "http://engine:8001")
    monkeypatch.setattr(oracle_api.requests, "post", fake_post)

    result = oracle_api.query_oracle(OracleQuery(question="linkedin", count=2))
    result_ids = [card.id for card in result.cards]

    assert result_ids == ["card-02", "card-04"]
    assert result.explanation == "Engine-selected cards"


def test_api_key_guard_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEY", "secret-key")
    with pytest.raises(HTTPException):
        oracle_api._require_api_key(None)
    oracle_api._require_api_key("secret-key")


def test_engine_stub_generate_is_deterministic() -> None:
    payload = llm_engine_stub.GenerateRequest(question="linkedin/github", count=3)
    first = llm_engine_stub.generate(payload)
    second = llm_engine_stub.generate(payload)
    assert first.card_ids == second.card_ids
    assert len(first.card_ids) == 3
