"""Transcript data model.

One shape for every source. Whether a transcript came from Whisper or was
uploaded as text, everything downstream sees the same object, so the
chunker, extractor and UI never branch on provenance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Iterable, Iterator

TIMESTAMP_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})$")

UNKNOWN_SPEAKER = "UNKNOWN"


class TranscriptFormatError(ValueError):
    """Raised when transcript text does not match the canonical format."""


def format_timestamp(seconds: float) -> str:
    """Seconds -> `HH:MM:SS`. Clamps negatives to zero."""
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def parse_timestamp(value: str) -> int:
    """`HH:MM:SS` -> seconds."""
    match = TIMESTAMP_RE.match(value.strip())
    if not match:
        raise TranscriptFormatError(f"bad timestamp {value!r}, expected HH:MM:SS")
    hours, minutes, secs = (int(g) for g in match.groups())
    if minutes > 59 or secs > 59:
        raise TranscriptFormatError(f"bad timestamp {value!r}, minutes/seconds out of range")
    return hours * 3600 + minutes * 60 + secs


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """One utterance. `start`/`end` are seconds from the start of the audio."""

    start: float
    end: float
    text: str
    speaker: str = UNKNOWN_SPEAKER

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise TranscriptFormatError(
                f"segment ends before it starts: {self.start} -> {self.end}"
            )

    @property
    def timestamp(self) -> str:
        return format_timestamp(self.start)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_line(self) -> str:
        return f"[{self.timestamp}] {self.speaker}: {self.text}"

    def with_speaker(self, speaker: str) -> TranscriptSegment:
        return replace(self, speaker=speaker)


@dataclass(frozen=True, slots=True)
class Transcript:
    """An ordered set of segments plus where they came from."""

    meeting_id: str
    segments: tuple[TranscriptSegment, ...] = field(default_factory=tuple)
    source: str = "text"  # "audio" | "text"
    language: str | None = None
    model: str | None = None

    def __iter__(self) -> Iterator[TranscriptSegment]:
        return iter(self.segments)

    def __len__(self) -> int:
        return len(self.segments)

    @property
    def duration(self) -> float:
        return self.segments[-1].end if self.segments else 0.0

    @property
    def speakers(self) -> tuple[str, ...]:
        """Distinct speakers in order of first appearance."""
        seen: dict[str, None] = {}
        for segment in self.segments:
            seen.setdefault(segment.speaker, None)
        return tuple(seen)

    @property
    def word_count(self) -> int:
        return sum(len(segment.text.split()) for segment in self.segments)

    def to_canonical_text(self) -> str:
        """Render as `[HH:MM:SS] Speaker: text`, one utterance per line.

        This is the on-disk interchange format and the exact shape the
        hand-labeled ground truth was written against.
        """
        return "\n".join(segment.to_line() for segment in self.segments) + "\n"

    def with_segments(self, segments: Iterable[TranscriptSegment]) -> Transcript:
        return replace(self, segments=tuple(segments))

    def segment_at(self, timestamp: str) -> TranscriptSegment | None:
        """Look up the segment starting at `HH:MM:SS`.

        Used to verify that an extracted item cites a real transcript span.
        """
        target = parse_timestamp(timestamp)
        for segment in self.segments:
            if int(segment.start) == target:
                return segment
        return None
