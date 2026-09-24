"""Provider-agnostic LLM interface.

One `complete()` call, two implementations. Swapping Groq for Ollama is an
env var, not a code change - which is both a vendor-lock-in argument and
the reason the pipeline still runs on a plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class LLMError(RuntimeError):
    """Raised when a provider is unavailable or returns nothing usable."""


class LLMUnavailableError(LLMError):
    """Raised when the provider cannot be reached or is not configured.

    Distinct from LLMError so callers can fall back to a deterministic
    path instead of failing outright.
    """


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    model: str
    provider: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@runtime_checkable
class LLMClient(Protocol):
    """What the rest of the codebase is allowed to assume about a provider."""

    name: str
    model: str

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> LLMResponse: ...
