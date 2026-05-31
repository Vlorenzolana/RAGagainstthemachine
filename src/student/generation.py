

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import List, Optional, Sequence

from student.models import MinimalSource

DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
DEFAULT_MAX_CONTEXT_CHARS = 8000
DEFAULT_MAX_NEW_TOKENS = 384

_SYSTEM_PROMPT = (
    "You are a precise assistant answering questions about the vLLM codebase. "
    "Use ONLY the provided context snippets to answer. "
    "If the answer is not in the context, say so briefly. "
    "Be concise, self-contained, and ground your answer in the sources by "
    "referring to file paths when relevant. Do not invent APIs."
)


@dataclass
class ContextSnippet:
    """A snippet shown to the LLM, sourced from a retrieved chunk."""

    file_path: str
    first_character_index: int
    last_character_index: int
    text: str


class AnswerGenerator:
    """Wraps a HuggingFace causal LM and produces grounded answers."""

    _lock = threading.Lock()

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        enable_thinking: bool = False,
        temperature: float = 0.0,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.max_context_chars = max_context_chars
        self.enable_thinking = enable_thinking
        self.temperature = temperature
        self._tokenizer: object | None = None
        self._model: object | None = None
        self._device: str = "cpu"

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        with self._lock:
            if self._model is not None and self._tokenizer is not None:
                return
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - defensive
                raise RuntimeError(
                    "transformers and torch are required for answer generation"
                ) from exc

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if self._device == "cuda" else torch.float32
            offline = os.environ.get("HF_HUB_OFFLINE") == "1"
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=True, local_files_only=offline
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                torch_dtype=dtype,
                local_files_only=offline,
            ).to(self._device)
            self._model.eval()  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
    ) -> str:
        """Generate an answer for ``question`` grounded in ``snippets``."""
        self._ensure_loaded()
        assert self._tokenizer is not None and self._model is not None
        import torch  # local import keeps the top-level deps optional

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
        try:
            prompt = self._tokenizer.apply_chat_template(  # type: ignore[attr-defined]
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            prompt = self._tokenizer.apply_chat_template(  # type: ignore[attr-defined]
                messages, tokenize=False, add_generation_prompt=True
            )

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)  # type: ignore[attr-defined]
        do_sample = self.temperature > 0.0
        with torch.no_grad():
            outputs = self._model.generate(  # type: ignore[attr-defined]
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else 1.0,
                top_p=0.95 if do_sample else 1.0,
                pad_token_id=getattr(self._tokenizer, "eos_token_id", None),
            )
        prompt_len = inputs["input_ids"].shape[1]
        generated = outputs[0][prompt_len:]
        text = self._tokenizer.decode(  # type: ignore[attr-defined]
            generated, skip_special_tokens=True
        ).strip()
        return _strip_thinking(text)


def _strip_thinking(text: str) -> str:
    """Remove any <think>...</think> blocks Qwen3 may emit."""
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>", start) + len("</think>")
        text = (text[:start] + text[end:]).strip()
    return text.strip()
