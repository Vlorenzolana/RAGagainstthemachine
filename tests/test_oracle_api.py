from __future__ import annotations

from fastapi.testclient import TestClient

import student.oracle_api as oracle_api
from student.generation import OpenRouterError
from student.models import Chunk


class FakeOpenRouterClient:
    def __init__(self, models: list[object] | None = None, fail: bool = False) -> None:
        self.models = models or []
        self.fail = fail
        self.calls = 0

    def list_models(self) -> list[object]:
        self.calls += 1
        if self.fail:
            raise OpenRouterError('upstream down')
        return self.models


def _reset_model_cache() -> None:
    oracle_api._model_cache = []
    oracle_api._model_cache_timestamp = 0.0


def test_models_endpoint_uses_cache(monkeypatch) -> None:
    _reset_model_cache()
    fake_client = FakeOpenRouterClient(
        models=[
            type('M', (), {
                'id': 'openai/gpt-4o-mini',
                'name': 'GPT-4o mini',
                'context_length': 128000,
                'prompt_price': '0.1',
                'completion_price': '0.2',
            })()
        ]
    )
    monkeypatch.setattr(oracle_api, '_get_client', lambda: fake_client)

    client = TestClient(oracle_api.app)
    first = client.get('/api/models')
    second = client.get('/api/models')

    assert first.status_code == 200
    assert second.status_code == 200
    assert fake_client.calls == 1
    assert first.json()[0]['id'] == 'openai/gpt-4o-mini'


def test_models_endpoint_returns_502_on_openrouter_error(monkeypatch) -> None:
    _reset_model_cache()
    fake_client = FakeOpenRouterClient(fail=True)
    monkeypatch.setattr(oracle_api, '_get_client', lambda: fake_client)

    client = TestClient(oracle_api.app)
    response = client.get('/api/models')

    assert response.status_code == 502
    assert 'Failed to fetch OpenRouter models' in response.json()['detail']


def test_ask_endpoint_uses_selected_model(monkeypatch) -> None:
    _reset_model_cache()

    class FakeRetriever:
        def __init__(self, _processed_dir) -> None:
            pass

        def search(self, _query: str, k: int = 10):
            del k
            return [
                (
                    Chunk(
                        chunk_id=1,
                        file_path='file.py',
                        first_character_index=0,
                        last_character_index=10,
                        text='chunk text',
                        kind='text',
                    ),
                    1.0,
                )
            ]

    captured: dict[str, str] = {}

    class FakeGenerator:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def generate(self, question, snippets, sources, model=None):
            del question, snippets, sources
            captured['model'] = model
            return 'ok'

    fake_client = FakeOpenRouterClient(
        models=[
            type('M', (), {
                'id': 'openai/gpt-4o-mini',
                'name': 'GPT-4o mini',
                'context_length': 128000,
                'prompt_price': '0.1',
                'completion_price': '0.2',
            })()
        ]
    )

    monkeypatch.setattr(oracle_api, '_get_client', lambda: fake_client)
    monkeypatch.setattr(oracle_api, 'BM25Retriever', FakeRetriever)
    monkeypatch.setattr(oracle_api, 'AnswerGenerator', FakeGenerator)

    client = TestClient(oracle_api.app)
    response = client.post(
        '/api/oracle/ask',
        json={'question': 'What?', 'k': 1, 'model': 'openai/gpt-4o-mini'},
    )

    assert response.status_code == 200
    body = response.json()
    assert body['answer'] == 'ok'
    assert body['model'] == 'openai/gpt-4o-mini'
    assert captured['model'] == 'openai/gpt-4o-mini'


def test_ask_endpoint_rejects_unknown_model(monkeypatch) -> None:
    _reset_model_cache()
    fake_client = FakeOpenRouterClient(
        models=[
            type('M', (), {
                'id': 'openai/gpt-4o-mini',
                'name': 'GPT-4o mini',
                'context_length': 128000,
                'prompt_price': '0.1',
                'completion_price': '0.2',
            })()
        ]
    )
    monkeypatch.setattr(oracle_api, '_get_client', lambda: fake_client)

    client = TestClient(oracle_api.app)
    response = client.post(
        '/api/oracle/ask',
        json={'question': 'What?', 'k': 1, 'model': 'unknown/vendor-model'},
    )

    assert response.status_code == 400
    assert response.json()['detail'] == 'Unsupported model id'
