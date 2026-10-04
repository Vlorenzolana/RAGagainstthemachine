from __future__ import annotations

import httpx
import pytest

from student.generation import (
    AnswerGenerator,
    ContextSnippet,
    OpenRouterClient,
    OpenRouterError,
)


def test_openrouter_chat_completion_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith('/chat/completions')
        payload = {
            'choices': [
                {'message': {'content': 'respuesta'}}
            ]
        }
        return httpx.Response(200, json=payload)

    client = OpenRouterClient(
        api_key='test-key',
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    text = client.create_chat_completion(
        model='openai/gpt-4o-mini',
        messages=[{'role': 'user', 'content': 'hola'}],
        max_tokens=64,
        temperature=0.0,
    )
    assert text == 'respuesta'


def test_openrouter_list_models_filters_non_text_models() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith('/models')
        return httpx.Response(
            200,
            json={
                'data': [
                    {
                        'id': 'openai/gpt-4o-mini',
                        'name': 'GPT 4o mini',
                        'context_length': 128000,
                        'architecture': {
                            'input_modalities': ['text'],
                            'output_modalities': ['text'],
                        },
                        'pricing': {'prompt': '0.1', 'completion': '0.2'},
                    },
                    {
                        'id': 'vision/only',
                        'name': 'Vision only',
                        'architecture': {
                            'input_modalities': ['image'],
                            'output_modalities': ['image'],
                        },
                    },
                ]
            },
        )

    client = OpenRouterClient(
        api_key='test-key',
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    models = client.list_models()
    assert [m.id for m in models] == ['openai/gpt-4o-mini']


def test_answer_generator_rejects_invalid_model_id() -> None:
    class DummyClient:
        def create_chat_completion(self, **kwargs: object) -> str:
            raise AssertionError('should not be called')

    generator = AnswerGenerator(client=DummyClient())
    with pytest.raises(OpenRouterError, match='Invalid OpenRouter model identifier'):
        generator.generate(
            'q',
            [ContextSnippet('a.py', 0, 1, 'x')],
            model='https://evil.example/model',
        )
