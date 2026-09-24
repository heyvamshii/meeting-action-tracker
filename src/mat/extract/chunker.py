"""Split a transcript into overlapping windows.

A 40-minute meeting is roughly 5,000 tokens and does not extract reliably
in one call - quality degrades long before the context limit is reached.
Windows overlap so an exchange that straddles a boundary is seen whole by
at least one chunk; the duplicates that creates are removed in merge.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from mat.config import settings
from mat.ingest import Transcript, TranscriptSegment

# Whisper-style English averages ~1.33 tokens per word. Good enough for
# sizing windows, and it avoids a tokenizer dependency for one number.
TOKENS_PER_WORD = 4 / 3


def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * TOKENS_PER_WORD)


@dataclass(frozen=True, slots=True)
class Chunk:
    index: int
    segments: tuple[TranscriptSegment, ...]
    overlap_count: int = 0  # leading segments repeated from the previous chunk

    @property
    def start_timestamp(self) -> str:
        return self.segments[0].timestamp

    @property
    def end_timestamp(self) -> str:
        return self.segments[-1].timestamp

    @property
    def token_estimate(self) -> int:
        return sum(estimate_tokens(s.text) for s in self.segments)

    def to_text(self) -> str:
        return "\n".join(s.to_line() for s in self.segments)


def chunk_transcript(
    transcript: Transcript,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[Chunk]:
    """Split into overlapping windows, never mid-utterance.

    A single segment longer than `max_tokens` becomes its own chunk rather
    than being cut - splitting an utterance would strand the speaker label
    and the timestamp from the text they belong to.
    """
    budget = max_tokens or settings.chunk_tokens
    overlap = overlap_tokens or settings.chunk_overlap_tokens

    if budget <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap >= budget:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    segments = list(transcript.segments)
    if not segments:
        return []

    chunks: list[Chunk] = []
    start = 0
    carried = 0

    while start < len(segments):
        used = 0
        end = start

        while end < len(segments):
            cost = estimate_tokens(segments[end].text)
            if used and used + cost > budget:
                break
            used += cost
            end += 1

        chunks.append(
            Chunk(
                index=len(chunks),
                segments=tuple(segments[start:end]),
                overlap_count=carried,
            )
        )

        if end >= len(segments):
            break

        next_start, carried = _overlap_start(segments, end, overlap)
        # Always advance, or an oversized segment loops forever.
        start = max(next_start, start + 1)

    return chunks


def _overlap_start(
    segments: list[TranscriptSegment], end: int, overlap_tokens: int
) -> tuple[int, int]:
    """Walk back from `end` until `overlap_tokens` worth of text is covered.

    Always carries at least one segment. Long utterances can each exceed the
    overlap budget on their own, and a strict budget would then produce zero
    overlap - silently removing the very protection overlap exists to give,
    which is that an exchange spanning a boundary is seen whole somewhere.
    """
    used = 0
    index = end

    while index > 0:
        cost = estimate_tokens(segments[index - 1].text)
        if used and used + cost > overlap_tokens:
            break
        used += cost
        index -= 1

    return index, end - index
