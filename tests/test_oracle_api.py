"""Focused tests for the lightweight Oracle API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from student import oracle_api


def _write_catalog(path: Path) -> None:
    cards = [
        {
            "id": f"card-{idx:02d}",
            "url": f"https://example.test/full/{idx}",
            "thumbnail": f"https://example.test/thumb/{idx}",
            "content_type": "image/jpeg",
        }
        for idx in range(1, 4)
    ]
    path.write_text(json.dumps(cards), encoding="utf-8")


def _client(monkeypatch: Any, tmp_path: Path) -> TestClient:
    catalog = tmp_path / "image_catalog.json"
    _write_catalog(catalog)
    monkeypatch.setattr(oracle_api, "CATALOG_PATH", catalog)
    monkeypatch.setattr(oracle_api, "API_KEY", None)
    monkeypatch.setattr(oracle_api, "LLM_ENGINE_URL", None)
    return TestClient(oracle_api.app)


def test_cards_count_validation(monkeypatch: Any, tmp_path: Path) -> None:
    client = _client(monkeypatch, tmp_path)
    response = client.get("/api/oracle/cards?count=0")
    assert response.status_code == 422


def test_query_fallback_is_deterministic(monkeypatch: Any, tmp_path: Path) -> None:
    client = _client(monkeypatch, tmp_path)
    payload = {"question": "¿Qué energía hay hoy?", "count": 2}

    response_a = client.post("/api/oracle/query", json=payload)
    response_b = client.post("/api/oracle/query", json=payload)

    assert response_a.status_code == 200
    assert response_b.status_code == 200
    data_a = response_a.json()
    data_b = response_b.json()
    assert [card["id"] for card in data_a["cards"]] == [card["id"] for card in data_b["cards"]]


def test_query_maps_engine_ids_to_catalog(monkeypatch: Any, tmp_path: Path) -> None:
    client = _client(monkeypatch, tmp_path)
    monkeypatch.setattr(
        oracle_api,
        "call_engine",
        lambda _question, _count: {
            "cards": ["card-03", "card-01"],
            "explanation": "Motor externo",
        },
    )
    response = client.post("/api/oracle/query", json={"question": "x", "count": 2})
    assert response.status_code == 200
    data = response.json()
    assert data["explanation"] == "Motor externo"
    assert [card["id"] for card in data["cards"]] == ["card-01", "card-03"]


def test_api_key_is_enforced_when_set(monkeypatch: Any, tmp_path: Path) -> None:
    client = _client(monkeypatch, tmp_path)
    monkeypatch.setattr(oracle_api, "API_KEY", "oracle-secret")

    unauthorized = client.get("/api/oracle/cards")
    authorized = client.get("/api/oracle/cards", headers={"x-api-key": "oracle-secret"})

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
