from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import httpx

from student.models import MinimalSource

DEFAULT_MODEL_ID = "openai/gpt-4o-mini"
DEFAULT_MAX_CONTEXT_CHARS = 8000
DEFAULT_MAX_NEW_TOKENS = 384
DEFAULT_TIMEOUT_SECONDS = 20.0

_SYSTEM_PROMPT = (
    "You are a precise assistant answering questions about the vLLM codebase. "
    "Use ONLY the provided context snippets to answer. "
    "If the answer is not in the context, say so briefly. "
    "Be concise, self-contained, and ground your answer in the sources by "
    "referring to file paths when relevant. Do not invent APIs."
)

_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9._:-]+/[A-Za-z0-9._:-]+$")


class OpenRouterError(RuntimeError):
    """Raised when OpenRouter request/response handling fails."""


@dataclass
class OpenRouterModel:
    id: str
    name: str
    context_length: int | None
    prompt_price: str | None = None
    completion_price: str | None = None


@dataclass
class ContextSnippet:
    """A snippet shown to the LLM, sourced from a retrieved chunk."""

    file_path: str
    first_character_index: int
    last_character_index: int
    text: str


class OpenRouterClient:
    """Small OpenRouter chat-completions client."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise OpenRouterError(
                "OPENROUTER_API_KEY is required for answer generation"
            )
        self.base_url = (base_url or os.environ.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")
        self._client = http_client or httpx.Client(timeout=timeout_seconds)

    def list_models(self) -> List[OpenRouterModel]:
        payload = self._request("GET", "/models")
        items = payload.get("data")
        if not isinstance(items, list):
            raise OpenRouterError("OpenRouter /models response is missing a data array")

        models: List[OpenRouterModel] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            model_id = item.get("id")
            if not isinstance(model_id, str) or not _is_valid_model_id(model_id):
                continue
            architecture = item.get("architecture") if isinstance(item.get("architecture"), dict) else {}
            input_modalities = architecture.get("input_modalities")
            output_modalities = architecture.get("output_modalities")
            if isinstance(input_modalities, list) and "text" not in input_modalities:
                continue
            if isinstance(output_modalities, list) and "text" not in output_modalities:
                continue

            pricing = item.get("pricing") if isinstance(item.get("pricing"), dict) else {}
            context_length = item.get("context_length")
            models.append(
                OpenRouterModel(
                    id=model_id,
                    name=str(item.get("name") or model_id),
                    context_length=int(context_length) if isinstance(context_length, int) else None,
                    prompt_price=_safe_str(pricing.get("prompt")),
                    completion_price=_safe_str(pricing.get("completion")),
                )
            )
        return models

    def create_chat_completion(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        payload = self._request(
            "POST",
            "/chat/completions",
            json={
                "model": model,
                "messages": list(messages),
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise OpenRouterError("OpenRouter response has no choices")
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first.get("message"), dict) else {}
        content = message.get("content")
        text = _content_to_text(content)
        if not text:
            raise OpenRouterError("OpenRouter response has empty content")
        return text.strip()

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers = {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}{path}"
        try:
            response = self._client.request(method, url, headers=headers, json=json)
        except httpx.TimeoutException as exc:
            raise OpenRouterError("OpenRouter request timed out") from exc
        except httpx.HTTPError as exc:
            raise OpenRouterError(f"OpenRouter request failed: {exc}") from exc

        if response.status_code >= 400:
            detail = _extract_error_message(response)
            raise OpenRouterError(
                f"OpenRouter error {response.status_code}: {detail}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise OpenRouterError("OpenRouter response is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise OpenRouterError("OpenRouter response payload has invalid shape")
        return payload


class AnswerGenerator:
    """Produces grounded answers using OpenRouter chat completions."""

    def __init__(
        self,
        model_id: str | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        enable_thinking: bool = False,
        temperature: float = 0.0,
        client: OpenRouterClient | None = None,
    ) -> None:
        self.model_id = model_id or os.environ.get("OPENROUTER_DEFAULT_MODEL", DEFAULT_MODEL_ID)
        self.max_new_tokens = max_new_tokens
        self.max_context_chars = max_context_chars
        self.enable_thinking = enable_thinking
        self.temperature = temperature
        self._client = client or OpenRouterClient()

    def build_context(self, snippets: Sequence[ContextSnippet]) -> str:
        """Concatenate snippets into a single context block within budget."""
        parts: List[str] = []
        total = 0
        for i, s in enumerate(snippets):
            header = (
                f"\n--- Source {i + 1}: {s.file_path}"
                f" [chars {s.first_character_index}-{s.last_character_index}] ---\n"
            )
            body = s.text
            remaining = self.max_context_chars - total - len(header)
            if remaining <= 0:
                break
            if len(body) > remaining:
                body = body[:remaining]
            parts.append(header + body)
            total += len(header) + len(body)
        return "".join(parts).strip()

    def generate(
        self,
        question: str,
        snippets: Sequence[ContextSnippet],
        sources: Optional[Sequence[MinimalSource]] = None,
        model: str | None = None,
    ) -> str:
        """Generate an answer for ``question`` grounded in ``snippets``."""
        del sources  # kept for compatibility with previous signature
        selected_model = model or self.model_id
        if not _is_valid_model_id(selected_model):
            raise OpenRouterError("Invalid OpenRouter model identifier")

        context = self.build_context(snippets)
        if not context:
            context = "(no relevant context retrieved)"
        user_msg = (
            f"Question: {question}\n\n"
            f"Context:\n{context}\n\n"
            "Answer the question using only the context above."
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        text = self._client.create_chat_completion(
            model=selected_model,
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return _strip_thinking(text)


def _is_valid_model_id(model_id: str) -> bool:
    return bool(_MODEL_ID_RE.match(model_id)) and not model_id.startswith("http")


def _safe_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _extract_error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or "unknown error"
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            message = err.get("message")
            if isinstance(message, str):
                return message
        if isinstance(err, str):
            return err
    return response.text.strip() or "unknown error"


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _strip_thinking(text: str) -> str:
    """Remove any <think>...</think> blocks a model may emit."""
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>", start) + len("</think>")
        text = (text[:start] + text[end:]).strip()
    return text.strip()
