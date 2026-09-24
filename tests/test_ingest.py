"""Phase 1: transcript model, parser, and audio-file validation.

Whisper itself is not exercised here - loading a model is slow and the
weights are not in the repo. What is tested is everything that can silently
corrupt downstream work: timestamp handling, round-tripping, and the
error paths a user will actually hit.
"""

from __future__ import annotations

import pytest

from mat.config import TRANSCRIPT_DIR
from mat.ingest import (
    AudioIngestError,
    Transcript,
    TranscriptFormatError,
    TranscriptSegment,
    format_timestamp,
    load_transcript_file,
    parse_timestamp,
    parse_transcript_text,
    save_transcript,
    validate_audio_file,
)

SAMPLE = """\
[00:00:04] Ravi (EM): Morning everyone.
[00:00:14] Deepa (Dev): I finished the retry logic.
[00:01:02] Ravi (EM): I'll chase Nisha for the email copy today.
"""


# --- timestamps ---------------------------------------------------------


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "00:00:00"), (5, "00:00:05"), (62, "00:01:02"), (3661, "01:01:01"), (-3, "00:00:00")],
)
def test_format_timestamp(seconds: int, expected: str) -> None:
    assert format_timestamp(seconds) == expected


def test_timestamp_round_trip() -> None:
    for seconds in (0, 59, 60, 599, 3599, 3600, 7322):
        assert parse_timestamp(format_timestamp(seconds)) == seconds


@pytest.mark.parametrize("value", ["1:02:03", "00:60:00", "00:00:60", "abc", ""])
def test_parse_timestamp_rejects_bad_input(value: str) -> None:
    with pytest.raises(TranscriptFormatError):
        parse_timestamp(value)


# --- segments -----------------------------------------------------------


def test_segment_line_format() -> None:
    segment = TranscriptSegment(start=62.4, end=70.0, text="Hello.", speaker="Ravi (EM)")
    assert segment.to_line() == "[00:01:02] Ravi (EM): Hello."
    assert segment.timestamp == "00:01:02"
    assert segment.duration == pytest.approx(7.6)


def test_segment_rejects_reversed_bounds() -> None:
    with pytest.raises(TranscriptFormatError):
        TranscriptSegment(start=10.0, end=4.0, text="x")


# --- parsing ------------------------------------------------------------


def test_parse_basic() -> None:
    transcript = parse_transcript_text(SAMPLE, meeting_id="demo")

    assert len(transcript) == 3
    assert transcript.meeting_id == "demo"
    assert transcript.source == "text"
    assert transcript.speakers == ("Ravi (EM)", "Deepa (Dev)")
    assert transcript.segments[0].text == "Morning everyone."


def test_segment_end_is_next_segment_start() -> None:
    """Text transcripts carry no end time; a segment runs until the next."""
    transcript = parse_transcript_text(SAMPLE, meeting_id="demo")
    assert transcript.segments[0].end == transcript.segments[1].start
    assert transcript.segments[-1].end > transcript.segments[-1].start


def test_parse_ignores_blank_lines() -> None:
    assert len(parse_transcript_text(SAMPLE.replace("\n", "\n\n"), meeting_id="demo")) == 3


def test_parse_tolerates_extra_whitespace() -> None:
    text = "[00:00:04]   Ravi (EM)  :   Morning everyone.  "
    segment = parse_transcript_text(text, meeting_id="demo").segments[0]
    assert segment.speaker == "Ravi (EM)"
    assert segment.text == "Morning everyone."


def test_parse_reports_the_offending_line_number() -> None:
    broken = SAMPLE.replace("[00:01:02] Ravi (EM):", "Ravi said")
    with pytest.raises(TranscriptFormatError, match="line 3"):
        parse_transcript_text(broken, meeting_id="demo")


def test_parse_rejects_backwards_timestamps() -> None:
    text = "[00:01:00] A: one\n[00:00:30] B: two\n"
    with pytest.raises(TranscriptFormatError, match="backwards"):
        parse_transcript_text(text, meeting_id="demo")


@pytest.mark.parametrize("text", ["", "   \n\n  "])
def test_parse_rejects_empty_transcript(text: str) -> None:
    with pytest.raises(TranscriptFormatError, match="empty"):
        parse_transcript_text(text, meeting_id="demo")


# --- round trip ---------------------------------------------------------


def test_canonical_text_round_trips() -> None:
    once = parse_transcript_text(SAMPLE, meeting_id="demo")
    twice = parse_transcript_text(once.to_canonical_text(), meeting_id="demo")
    assert once.to_canonical_text() == twice.to_canonical_text()


def test_corpus_files_round_trip_unchanged() -> None:
    """The committed corpus must survive a parse/render cycle byte for byte.

    Ground truth timestamps are cited against these exact files.
    """
    for path in sorted(TRANSCRIPT_DIR.glob("*.txt")):
        transcript = load_transcript_file(path)
        assert transcript.to_canonical_text() == path.read_text(encoding="utf-8"), path.name


def test_save_and_reload(tmp_path) -> None:
    original = parse_transcript_text(SAMPLE, meeting_id="demo")
    destination = save_transcript(original, tmp_path / "nested" / "demo.txt")

    assert destination.exists()
    assert load_transcript_file(destination).to_canonical_text() == original.to_canonical_text()


# --- lookup used by the extraction guardrail ----------------------------


def test_segment_at_finds_cited_timestamp() -> None:
    transcript = parse_transcript_text(SAMPLE, meeting_id="demo")
    found = transcript.segment_at("00:01:02")
    assert found is not None and "Nisha" in found.text


def test_segment_at_returns_none_for_invented_timestamp() -> None:
    transcript = parse_transcript_text(SAMPLE, meeting_id="demo")
    assert transcript.segment_at("00:09:99".replace("99", "59")) is None


# --- audio validation ---------------------------------------------------


def test_validate_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(AudioIngestError, match="no such file"):
        validate_audio_file(tmp_path / "nope.wav")


def test_validate_rejects_unsupported_extension(tmp_path) -> None:
    path = tmp_path / "notes.pdf"
    path.write_bytes(b"x")
    with pytest.raises(AudioIngestError, match="unsupported format"):
        validate_audio_file(path)


def test_validate_rejects_empty_file(tmp_path) -> None:
    path = tmp_path / "silence.wav"
    path.touch()
    with pytest.raises(AudioIngestError, match="empty"):
        validate_audio_file(path)


def test_validate_accepts_supported_audio(tmp_path) -> None:
    path = tmp_path / "meeting.M4A"
    path.write_bytes(b"not really audio, but validation is extension + size")
    assert validate_audio_file(path) == path


# --- empty transcript ---------------------------------------------------


def test_empty_transcript_has_zero_duration() -> None:
    empty = Transcript(meeting_id="demo")
    assert len(empty) == 0
    assert empty.duration == 0.0
    assert empty.speakers == ()
