"""Phase 0 guard: transcripts must parse before anything downstream runs.

The extractor, the chunker and the UI's click-to-jump all assume the
`[HH:MM:SS] Speaker: text` shape. Breaking it here breaks all three.
"""

from __future__ import annotations

import re

import pytest

from mat.config import TRANSCRIPT_DIR

LINE_RE = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\] ([^:]+): (.+)$")


def _transcripts() -> list:
    return sorted(TRANSCRIPT_DIR.glob("*.txt"))


def test_corpus_has_transcripts() -> None:
    assert _transcripts(), "no transcripts present"


@pytest.mark.parametrize("path", _transcripts(), ids=lambda p: p.stem)
def test_every_line_parses(path) -> None:
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        assert LINE_RE.match(line), f"{path.name}:{lineno} malformed: {line[:60]!r}"


@pytest.mark.parametrize("path", _transcripts(), ids=lambda p: p.stem)
def test_timestamps_are_monotonic(path) -> None:
    previous = -1
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = LINE_RE.match(line)
        if not match:
            continue
        h, m, s = (int(g) for g in match.groups()[:3])
        seconds = h * 3600 + m * 60 + s
        assert seconds >= previous, f"{path.name}:{lineno} timestamp goes backwards"
        previous = seconds


@pytest.mark.parametrize("path", _transcripts(), ids=lambda p: p.stem)
def test_speakers_are_consistently_named(path) -> None:
    """A speaker must be spelled the same way throughout, or owner
    attribution silently fragments across variants."""
    speakers = {
        match.group(4)
        for line in path.read_text(encoding="utf-8").splitlines()
        if (match := LINE_RE.match(line))
    }
    bare_names = [s.split(" (")[0].strip().lower() for s in speakers]
    assert len(bare_names) == len(set(bare_names)), (
        f"{path.name}: same person appears under multiple labels: {sorted(speakers)}"
    )
