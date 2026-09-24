"""Transcription via Groq's hosted Whisper.

Local `base` on CPU is fine for clean, close-mic speech. On a real meeting
recording - accented English, a laptop microphone, two people talking at
once - it degrades into fragments ("I'm." "And." "Sorry.") and the
extractor correctly finds nothing in them.

`whisper-large-v3` is the same family, several times larger, and runs on
Groq's hardware in seconds. It is the difference between a demo that works
on a real recording and one that only works on clean audio.

Local transcription stays the default fallback: no key, no network, no
data leaving the machine.
"""

from __future__ import annotations

from pathlib import Path

from mat.config import settings

from .compress import extract_audio, needs_compression
from .models import Transcript, TranscriptSegment, UNKNOWN_SPEAKER

# Groq rejects uploads above this on the free tier. Larger files have their
# audio track extracted first, which is usually a 20-30x reduction.
MAX_UPLOAD_BYTES = 24 * 1024 * 1024


class RemoteTranscriptionError(RuntimeError):
    """Raised when hosted transcription is unavailable or fails."""


def transcribe_remote(
    path: str | Path,
    meeting_id: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> tuple[Transcript, float, str]:
    """Transcribe with Groq Whisper. Returns (transcript, duration, language)."""
    path = Path(path)
    key = api_key or settings.groq_api_key
    if not key:
        raise RemoteTranscriptionError(
            "GROQ_API_KEY is not set; hosted transcription is unavailable"
        )

    try:
        from groq import Groq
    except ImportError as exc:  # pragma: no cover - environment problem
        raise RemoteTranscriptionError("the 'groq' package is not installed") from exc

    upload = path
    temporary = None
    if needs_compression(path, MAX_UPLOAD_BYTES):
        temporary = extract_audio(path)
        upload = temporary

    try:
        with upload.open("rb") as handle:
            response = Groq(api_key=key).audio.transcriptions.create(
                file=(upload.name, handle.read()),
                model=model or settings.groq_whisper_model,
                response_format="verbose_json",
                # Timestamps per utterance, not per word: the pipeline cites
                # lines, and word-level output would be an order of magnitude
                # more data for no gain.
                timestamp_granularities=["segment"],
            )
    except Exception as exc:  # noqa: BLE001 - SDK raises a wide range
        raise RemoteTranscriptionError(f"groq transcription failed: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    payload = response if isinstance(response, dict) else response.model_dump()
    segments = _to_segments(payload.get("segments") or [])

    if not segments:
        raise RemoteTranscriptionError(f"no speech detected in {path.name}")

    transcript = Transcript(
        meeting_id=meeting_id or path.stem,
        segments=segments,
        source="audio",
        language=payload.get("language"),
        model=model or settings.groq_whisper_model,
    )
    return transcript, float(payload.get("duration") or 0.0), payload.get("language") or ""


def _to_segments(raw: list[dict]) -> tuple[TranscriptSegment, ...]:
    segments = []
    for entry in raw:
        text = (entry.get("text") or "").strip()
        if not text:
            continue
        segments.append(
            TranscriptSegment(
                start=float(entry.get("start", 0.0)),
                end=float(entry.get("end", 0.0)),
                text=text,
                speaker=UNKNOWN_SPEAKER,
            )
        )
    return tuple(segments)
