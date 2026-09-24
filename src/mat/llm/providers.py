"""Groq and Ollama clients behind the same interface, plus the factory.

Groq is the default: free tier, fast enough that a 40-minute transcript
extracts in seconds. Ollama is the offline fallback - slower and weaker at
strict JSON, but it needs no network and no key.
"""

from __future__ import annotations

import time

from mat.config import settings

from .base import LLMClient, LLMResponse, LLMUnavailableError

# The quota is a fixed one-minute window, so waiting it out is the remedy.
RATE_LIMIT_WAIT_SECONDS = 62

RATE_LIMIT_MARKERS = ("rate_limit", "rate limit", "429", "tokens per minute", "tpm")


def _is_rate_limit(exc: Exception) -> bool:
    """True when the provider refused because a quota window is exhausted.

    Matched on the message rather than an exception type: Groq reports the
    token-per-minute ceiling as a 413 ("request too large") as well as a
    429, and both are cured by the same wait.
    """
    text = str(exc).lower()
    return any(marker in text for marker in RATE_LIMIT_MARKERS)


class GroqClient:
    name = "groq"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.model = model or settings.groq_model
        key = api_key or settings.groq_api_key
        if not key:
            raise LLMUnavailableError(
                "GROQ_API_KEY is not set. Add it to .env, or set LLM_PROVIDER=ollama."
            )
        try:
            from groq import Groq
        except ImportError as exc:  # pragma: no cover - environment problem
            raise LLMUnavailableError("the 'groq' package is not installed") from exc

        self._client = Groq(api_key=key)

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        # Sent via extra_body because groq 0.13.x rejects reasoning_effort as
        # a keyword argument. Models that do not understand it ignore it.
        if settings.groq_reasoning_effort:
            kwargs["extra_body"] = {"reasoning_effort": settings.groq_reasoning_effort}

        response = self._create_with_retry(kwargs)

        usage = getattr(response, "usage", None)
        return LLMResponse(
            text=response.choices[0].message.content or "",
            model=self.model,
            provider=self.name,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )

    def _create_with_retry(self, kwargs: dict):
        """Retry once per minute when the per-minute token budget is spent.

        Groq's free tier allows 8000 tokens per minute, and `max_tokens`
        counts toward a request's size. A long meeting extracted chunk by
        chunk will exhaust that mid-run: without this, chunks 2 and 3 of a
        40-minute meeting simply failed and two thirds of the meeting was
        silently missing from the board.

        The wait is deliberate rather than clever. The limit is per minute,
        so waiting out the window is the fix; exponential backoff on a
        fixed-window quota just fails faster.
        """
        last_error: Exception | None = None

        for attempt in range(settings.llm_rate_limit_retries + 1):
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001 - SDK raises a wide range
                last_error = exc
                if not _is_rate_limit(exc) or attempt == settings.llm_rate_limit_retries:
                    break
                time.sleep(RATE_LIMIT_WAIT_SECONDS)

        raise LLMUnavailableError(f"groq request failed: {last_error}") from last_error


class OllamaClient:
    name = "ollama"

    def __init__(self, host: str | None = None, model: str | None = None) -> None:
        self.model = model or settings.ollama_model
        try:
            import ollama
        except ImportError as exc:  # pragma: no cover - environment problem
            raise LLMUnavailableError("the 'ollama' package is not installed") from exc

        self._client = ollama.Client(host=host or settings.ollama_host)

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat(
                model=self.model,
                messages=messages,
                format="json" if json_mode else None,
                options={"temperature": temperature, "num_predict": max_tokens},
            )
        except Exception as exc:  # noqa: BLE001 - SDK raises a wide range
            raise LLMUnavailableError(
                f"ollama request failed ({settings.ollama_host}); is `ollama serve` running? {exc}"
            ) from exc

        return LLMResponse(
            text=response["message"]["content"],
            model=self.model,
            provider=self.name,
            prompt_tokens=response.get("prompt_eval_count"),
            completion_tokens=response.get("eval_count"),
        )


def get_client(provider: str | None = None, model: str | None = None) -> LLMClient:
    """Build the configured client. Raises LLMUnavailableError if it cannot."""
    choice = (provider or settings.llm_provider).strip().lower()

    if choice == "groq":
        return GroqClient(model=model)
    if choice == "ollama":
        return OllamaClient(model=model)

    raise LLMUnavailableError(f"unknown LLM_PROVIDER {choice!r}; expected 'groq' or 'ollama'")


def try_get_client(provider: str | None = None, model: str | None = None) -> LLMClient | None:
    """Same, but returns None instead of raising.

    Used where an LLM is an upgrade rather than a requirement - speaker
    attribution still works heuristically without one.
    """
    try:
        return get_client(provider, model)
    except LLMUnavailableError:
        return None
