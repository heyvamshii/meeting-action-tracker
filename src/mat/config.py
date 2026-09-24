"""Central config. Everything tunable lives here, read from .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"


def _env(key: str, default: str) -> str:
    return os.getenv(key, default) or default


@dataclass(frozen=True)
class Settings:
    llm_provider: str = _env("LLM_PROVIDER", "groq")
    groq_api_key: str = _env("GROQ_API_KEY", "")
    groq_model: str = _env("GROQ_MODEL", "openai/gpt-oss-120b")
    ollama_host: str = _env("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = _env("OLLAMA_MODEL", "llama3.1:8b")

    # local | groq. Hosted is far more accurate on real meeting audio;
    # local needs no key and no network.
    transcription_backend: str = _env("TRANSCRIPTION_BACKEND", "groq")
    groq_whisper_model: str = _env("GROQ_WHISPER_MODEL", "whisper-large-v3")
    whisper_model: str = _env("WHISPER_MODEL", "base")
    whisper_compute_type: str = _env("WHISPER_COMPUTE_TYPE", "int8")

    db_path: Path = PROJECT_ROOT / _env("DB_PATH", "data/db/meetings.db")

    # Counts toward the request size on Groq, whose free tier allows 8000
    # tokens per minute. prompt + this must stay under that ceiling.
    llm_max_tokens: int = int(_env("LLM_MAX_TOKENS", "3000"))
    # gpt-oss models are reasoning models: they spend output tokens thinking
    # before the first character of JSON. Measured on one extraction chunk,
    # "low" halved completion tokens (2140 -> 1058) with no loss of quality.
    # Empty string disables the parameter for models that reject it.
    groq_reasoning_effort: str = _env("GROQ_REASONING_EFFORT", "low")
    llm_rate_limit_retries: int = int(_env("LLM_RATE_LIMIT_RETRIES", "3"))

    # 800 keeps prompt+budget inside Groq's 8000 tokens-per-minute ceiling:
    # a chunk this size measured 2245 prompt tokens.
    chunk_tokens: int = int(_env("CHUNK_TOKENS", "800"))
    chunk_overlap_tokens: int = int(_env("CHUNK_OVERLAP_TOKENS", "150"))
    low_confidence_threshold: float = float(_env("LOW_CONFIDENCE_THRESHOLD", "0.80"))


settings = Settings()
