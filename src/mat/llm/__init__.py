"""LLM access layer. One interface, two providers, swapped by env var."""

from .base import LLMClient, LLMError, LLMResponse, LLMUnavailableError
from .parsing import JSONParseError, parse_json_object
from .providers import GroqClient, OllamaClient, get_client, try_get_client

__all__ = [
    "GroqClient",
    "JSONParseError",
    "LLMClient",
    "LLMError",
    "LLMResponse",
    "LLMUnavailableError",
    "OllamaClient",
    "get_client",
    "parse_json_object",
    "try_get_client",
]
