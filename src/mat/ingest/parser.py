"""Parse canonical transcript text into a Transcript.

The text path is first-class, not a fallback. If transcription is slow or
the model is unavailable, the whole pipeline still runs from a pasted
Zoom/Teams export.
"""

from __future__ import annotations

from pathlib import Path

import re

from .models import (
    Transcript,
    TranscriptFormatError,
    TranscriptSegment,
    parse_timestamp,
)

LINE_RE = re.compile(r"^\[(?P<ts>\d{2}:\d{2}:\d{2})\]\s*(?P<speaker>[^:]+?)\s*:\s*(?P<text>.+)$")

# A segment's end is unknown in the text format; assume it runs until the
# next one starts. The final segment gets this nominal length instead.
TRAILING_SEGMENT_SECONDS = 5.0


def parse_transcript_text(text: str, meeting_id: str) -> Transcript:
    """Parse `[HH:MM:SS] Speaker: text` lines.

    Raises TranscriptFormatError with a line number on the first bad line,
    so the UI can tell the user exactly what to fix.
    """
    starts: list[int] = []
    speakers: list[str] = []
    bodies: list[str] = []

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue

        match = LINE_RE.match(line)
        if not match:
            raise TranscriptFormatError(
                f"line {lineno}: expected '[HH:MM:SS] Speaker: text', got {line[:60]!r}"
            )

        start = parse_timestamp(match.group("ts"))
        if starts and start < starts[-1]:
            raise TranscriptFormatError(
                f"line {lineno}: timestamp {match.group('ts')} goes backwards"
            )

        starts.append(start)
        speakers.append(match.group("speaker").strip())
        bodies.append(match.group("text").strip())

    if not starts:
        raise TranscriptFormatError("transcript is empty")

    segments = tuple(
        TranscriptSegment(
            start=float(start),
            end=float(starts[i + 1]) if i + 1 < len(starts) else start + TRAILING_SEGMENT_SECONDS,
            text=bodies[i],
            speaker=speakers[i],
        )
        for i, start in enumerate(starts)
    )

    return Transcript(meeting_id=meeting_id, segments=segments, source="text")


def load_transcript_file(path: str | Path, meeting_id: str | None = None) -> Transcript:
    """Read a canonical transcript from disk. Filename stem is the default id."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no transcript at {path}")
    return parse_transcript_text(
        path.read_text(encoding="utf-8"), meeting_id=meeting_id or path.stem
    )


def save_transcript(transcript: Transcript, path: str | Path) -> Path:
    """Write a Transcript back out in canonical form."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(transcript.to_canonical_text(), encoding="utf-8")
    return path
