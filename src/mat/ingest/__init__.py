"""Ingest layer: audio or text in, a canonical Transcript out."""

from .audio import (
    AUDIO_SUFFIXES,
    AudioIngestError,
    SUPPORTED_SUFFIXES,
    VIDEO_SUFFIXES,
    TranscriptionResult,
    transcribe,
    validate_audio_file,
)
from .models import (
    Transcript,
    TranscriptFormatError,
    TranscriptSegment,
    UNKNOWN_SPEAKER,
    format_timestamp,
    parse_timestamp,
)
from .compress import CompressionError, extract_audio
from .parser import load_transcript_file, parse_transcript_text, save_transcript
from .remote import RemoteTranscriptionError, transcribe_remote
from .roster import Participant, Roster, split_label
from .speakers import AttributionResult, attribute_speakers, is_labeled

__all__ = [
    "AUDIO_SUFFIXES",
    "AttributionResult",
    "CompressionError",
    "AudioIngestError",
    "Participant",
    "Roster",
    "SUPPORTED_SUFFIXES",
    "Transcript",
    "TranscriptFormatError",
    "TranscriptSegment",
    "TranscriptionResult",
    "VIDEO_SUFFIXES",
    "UNKNOWN_SPEAKER",
    "attribute_speakers",
    "extract_audio",
    "format_timestamp",
    "is_labeled",
    "load_transcript_file",
    "parse_timestamp",
    "parse_transcript_text",
    "save_transcript",
    "split_label",
    "RemoteTranscriptionError",
    "transcribe",
    "transcribe_remote",
    "validate_audio_file",
]
