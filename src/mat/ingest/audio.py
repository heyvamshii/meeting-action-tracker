"""Audio -> timestamped transcript via faster-whisper on CPU.

Decoding goes through PyAV, which ships with faster-whisper, so no system
ffmpeg install is required.

The model is loaded lazily and cached per (name, compute_type): loading
`base` takes several seconds, and Streamlit re-runs the script on every
interaction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from mat.config import settings

from .models import Transcript, TranscriptSegment, UNKNOWN_SPEAKER
from .remote import RemoteTranscriptionError, transcribe_remote

AUDIO_SUFFIXES = frozenset({".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".opus"})

# Video works because PyAV demuxes the container and Whisper is handed the
# audio stream; the picture is simply never decoded. A screen recording is
# therefore a first-class input - no conversion step, no extra dependency.
VIDEO_SUFFIXES = frozenset({".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".wmv"})

SUPPORTED_SUFFIXES = AUDIO_SUFFIXES | VIDEO_SUFFIXES

# Above this, transcription on CPU takes long enough that the user should
# be warned rather than left staring at a spinner.
LONG_AUDIO_SECONDS = 45 * 60

# Conversational English runs 110-160 words per minute. Far below that means
# the model heard fragments rather than sentences, and the transcript is not
# worth extracting from. Measured on a real failure: 549 words in 4.7 minutes
# (117 wpm of "I'm." "And." "Sorry.") - so the bar is set low deliberately,
# to catch collapse rather than merely slow speech.
POOR_TRANSCRIPT_WPM = 90

# Screen recordings are large. Above this the upload, not the transcription,
# is usually what makes the UI feel broken.
LARGE_VIDEO_BYTES = 500 * 1024 * 1024

_MODEL_CACHE: dict[tuple[str, str], object] = {}

ProgressCallback = Callable[[float, str], None]


class AudioIngestError(RuntimeError):
    """Raised when audio cannot be turned into a usable transcript."""


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    transcript: Transcript
    duration: float
    language: str
    language_probability: float

    source_bytes: int = 0
    is_video: bool = False
    backend: str = "local"

    @property
    def words_per_minute(self) -> float:
        minutes = self.duration / 60
        return self.transcript.word_count / minutes if minutes else 0.0

    @property
    def warnings(self) -> tuple[str, ...]:
        issues = []
        if self.source_bytes > LARGE_VIDEO_BYTES:
            issues.append(
                f"source file is {self.source_bytes / 1_048_576:.0f} MB; the browser "
                "upload, not the transcription, is usually the slow part"
            )
        if self.duration > LONG_AUDIO_SECONDS:
            issues.append(
                f"audio is {self.duration / 60:.0f} min; transcription on CPU is slow "
                "and extraction will span many chunks"
            )
        if self.words_per_minute and self.words_per_minute < POOR_TRANSCRIPT_WPM:
            issues.append(
                f"only {self.words_per_minute:.0f} words per minute were recognised - "
                "the transcript is probably fragmented. Speech is normally 110-160. "
                "Try the hosted 'groq' backend or a larger local model; extraction "
                "from a broken transcript will find little or nothing"
            )
        if self.language_probability < 0.6:
            issues.append(
                f"language detected as '{self.language}' with low confidence "
                f"({self.language_probability:.2f}); output may be unreliable"
            )
        return tuple(issues)


def validate_audio_file(path: str | Path) -> Path:
    """Check the file exists, is non-empty and has a decodable extension."""
    path = Path(path)
    if not path.exists():
        raise AudioIngestError(f"no such file: {path}")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise AudioIngestError(f"unsupported format '{path.suffix}'. Supported: {supported}")
    if path.stat().st_size == 0:
        raise AudioIngestError(f"file is empty: {path.name}")
    return path


def load_model(model_name: str | None = None, compute_type: str | None = None):
    """Load (and cache) a faster-whisper model on CPU."""
    name = model_name or settings.whisper_model
    compute = compute_type or settings.whisper_compute_type
    key = (name, compute)

    if key not in _MODEL_CACHE:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - environment problem
            raise AudioIngestError(
                "faster-whisper is not installed; use the transcript upload path instead"
            ) from exc
        _MODEL_CACHE[key] = WhisperModel(name, device="cpu", compute_type=compute)

    return _MODEL_CACHE[key]


def transcribe(
    path: str | Path,
    meeting_id: str | None = None,
    model_name: str | None = None,
    on_progress: ProgressCallback | None = None,
    backend: str | None = None,
) -> TranscriptionResult:
    """Transcribe an audio or video file into a Transcript.

    `backend` is "groq" (hosted whisper-large-v3, far better on real
    meeting audio) or "local" (faster-whisper on CPU, no network). Hosted
    falls back to local automatically if it is unavailable, so a missing
    key or a dropped connection degrades rather than fails.

    `on_progress` receives (fraction_complete, latest_text) as segments
    arrive, so the UI and CLI can both show real movement instead of an
    indeterminate spinner.
    """
    path = validate_audio_file(path)

    choice = (backend or settings.transcription_backend).strip().lower()
    if choice == "groq":
        try:
            transcript, duration, language = transcribe_remote(path, meeting_id)
            if on_progress:
                on_progress(1.0, "done")
            return TranscriptionResult(
                transcript=transcript,
                duration=duration,
                language=language,
                language_probability=1.0,
                source_bytes=path.stat().st_size,
                is_video=path.suffix.lower() in VIDEO_SUFFIXES,
                backend="groq",
            )
        except RemoteTranscriptionError:
            # Fall through to local rather than failing the upload.
            pass

    model = load_model(model_name)

    raw_segments, info = model.transcribe(
        str(path),
        beam_size=5,
        # Silence is where Whisper invents text. Trimming it is the cheapest
        # hallucination guardrail available at this layer.
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    segments = tuple(_collect(raw_segments, info.duration, on_progress))
    if not segments:
        raise AudioIngestError(
            f"no speech detected in {path.name} (audio may be silent or music only)"
        )

    transcript = Transcript(
        meeting_id=meeting_id or path.stem,
        segments=segments,
        source="audio",
        language=info.language,
        model=model_name or settings.whisper_model,
    )

    return TranscriptionResult(
        transcript=transcript,
        duration=info.duration,
        language=info.language,
        language_probability=info.language_probability,
        source_bytes=path.stat().st_size,
        is_video=path.suffix.lower() in VIDEO_SUFFIXES,
        backend="local",
    )


def _collect(
    raw_segments,
    duration: float,
    on_progress: ProgressCallback | None,
) -> Iterator[TranscriptSegment]:
    """Drain the lazy segment generator, reporting progress as it goes."""
    for raw in raw_segments:
        text = raw.text.strip()
        if not text:
            continue

        segment = TranscriptSegment(
            start=raw.start,
            end=raw.end,
            text=text,
            speaker=UNKNOWN_SPEAKER,
        )

        if on_progress and duration > 0:
            on_progress(min(raw.end / duration, 1.0), text)

        yield segment
