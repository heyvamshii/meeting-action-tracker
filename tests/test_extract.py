"""Phase 3: schema guardrails, chunking, dedupe, and the engine.

The engine is driven by a stub client. What is under test is not whether a
model extracts well - that is Phase 7's job, measured against hand labels -
but whether bad model output can reach the database. It must not.
"""

from __future__ import annotations

import json
from datetime import date

import pytest
from pydantic import ValidationError

from mat.extract import (
    ActionItem,
    ChunkExtract,
    Decision,
    DueDateSource,
    ItemStatus,
    OwnerSource,
    UNASSIGNED,
    chunk_transcript,
    estimate_tokens,
    extract_meeting,
    merge_chunks,
    similarity,
)
from mat.extract.engine import _validate_items, ExtractionReport
from mat.ingest import parse_transcript_text
from mat.llm.base import LLMResponse

MEETING_DATE = date(2026, 9, 3)

TRANSCRIPT = """\
[00:00:04] Priya (PM): Where are we on the payment gateway?
[00:00:20] Arun (Dev): Sandbox is done, production keys are pending.
[00:00:35] Mark (Client): I'll get you the production keys by Friday.
[00:00:44] Priya (PM): So we're targeting the 15th, not the 10th. Agreed?
[00:00:52] Arun (Dev): Yes, that works.
[00:01:10] Sneha (QA): Do we need PCI sign-off before go-live?
"""


def transcript():
    return parse_transcript_text(TRANSCRIPT, meeting_id="demo")


class StubClient:
    """Replays canned payloads, one per call."""

    name = "stub"
    model = "stub-1"

    def __init__(self, *payloads) -> None:
        self._payloads = list(payloads)
        self.calls = 0

    def complete(self, prompt, system=None, temperature=0.0, max_tokens=4096, json_mode=False):
        self.calls += 1
        payload = self._payloads[min(self.calls - 1, len(self._payloads) - 1)]
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return LLMResponse(text=text, model=self.model, provider=self.name)


def action(**overrides):
    base = {
        "task": "Provide production keys",
        "owner": "Mark (Client)",
        "owner_source": "explicit",
        "due_phrase": "by Friday",
        "timestamp": "00:00:35",
        "confidence": 0.9,
    }
    return {**base, **overrides}


# --- schema guardrails --------------------------------------------------


def test_valid_action_item() -> None:
    item = ActionItem.model_validate(action())
    assert item.owner_source is OwnerSource.EXPLICIT
    assert item.status is ItemStatus.OPEN


@pytest.mark.parametrize("timestamp", ["1:02:03", "00:00", "midway", ""])
def test_bad_timestamp_is_rejected(timestamp) -> None:
    with pytest.raises(ValidationError):
        ActionItem.model_validate(action(timestamp=timestamp))


@pytest.mark.parametrize("confidence", [-0.1, 1.5])
def test_confidence_must_be_a_probability(confidence) -> None:
    with pytest.raises(ValidationError):
        ActionItem.model_validate(action(confidence=confidence))


def test_empty_task_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ActionItem.model_validate(action(task=""))


def test_owner_without_provenance_is_rejected() -> None:
    """An owner with source 'none' is exactly the confident guess the
    provenance field exists to expose."""
    with pytest.raises(ValidationError, match="owner_source"):
        ActionItem.model_validate(action(owner="Mark", owner_source="none"))


def test_provenance_without_owner_is_rejected() -> None:
    with pytest.raises(ValidationError, match="owner_source"):
        ActionItem.model_validate(action(owner=UNASSIGNED, owner_source="explicit"))


def test_date_without_provenance_is_rejected() -> None:
    with pytest.raises(ValidationError, match="due_date"):
        ActionItem.model_validate(
            action(due_date="2026-09-04", due_date_source="none")
        )


def test_unassigned_item_is_valid() -> None:
    item = ActionItem.model_validate(
        action(owner=UNASSIGNED, owner_source="none", due_phrase=None)
    )
    assert item.owner == UNASSIGNED
    assert item.due_date is None


def test_unknown_enum_value_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ActionItem.model_validate(action(status="maybe"))


# --- chunking -----------------------------------------------------------


def test_short_transcript_is_one_chunk() -> None:
    assert len(chunk_transcript(transcript(), max_tokens=5000, overlap_tokens=100)) == 1


def test_long_transcript_is_split_with_overlap() -> None:
    chunks = chunk_transcript(transcript(), max_tokens=15, overlap_tokens=5)

    assert len(chunks) > 1
    assert all(chunk.segments for chunk in chunks)
    assert any(chunk.overlap_count > 0 for chunk in chunks[1:])


def test_chunks_cover_every_segment() -> None:
    source = transcript()
    covered = {
        segment.timestamp
        for chunk in chunk_transcript(source, max_tokens=15, overlap_tokens=5)
        for segment in chunk.segments
    }
    assert covered == {segment.timestamp for segment in source}


def test_chunks_never_split_an_utterance() -> None:
    """Splitting mid-utterance would strand the text from its timestamp."""
    source = transcript()
    texts = {segment.text for segment in source}
    for chunk in chunk_transcript(source, max_tokens=5, overlap_tokens=1):
        for segment in chunk.segments:
            assert segment.text in texts


def test_oversized_segment_becomes_its_own_chunk() -> None:
    long_line = "[00:00:01] A: " + " ".join(["word"] * 500) + "\n"
    chunks = chunk_transcript(
        parse_transcript_text(long_line, meeting_id="m"), max_tokens=10, overlap_tokens=2
    )
    assert len(chunks) == 1


def test_empty_transcript_yields_no_chunks() -> None:
    from mat.ingest import Transcript

    assert chunk_transcript(Transcript(meeting_id="m")) == []


def test_overlap_must_be_smaller_than_window() -> None:
    with pytest.raises(ValueError, match="overlap"):
        chunk_transcript(transcript(), max_tokens=100, overlap_tokens=100)


def test_token_estimate_scales_with_words() -> None:
    assert estimate_tokens("one two three") < estimate_tokens("one two three four five six")


# --- dedupe -------------------------------------------------------------


def test_similarity_ignores_wording_and_stopwords() -> None:
    assert similarity("cap the exports at 500k rows", "add a 500k row cap to exports") > 0.5
    assert similarity("cap the exports", "write the pagination story") < 0.5


def test_overlap_duplicates_collapse_to_the_earliest() -> None:
    first = ChunkExtract(action_items=[ActionItem.model_validate(action())])
    second = ChunkExtract(
        action_items=[ActionItem.model_validate(action(timestamp="00:00:44", confidence=0.7))]
    )

    merged = merge_chunks([first, second])
    assert len(merged.action_items) == 1
    assert merged.action_items[0].timestamp == "00:00:35"


def test_recap_restatement_collapses_and_keeps_earliest() -> None:
    """A chair recapping commitments must not double every action item."""
    original = ChunkExtract(action_items=[ActionItem.model_validate(action())])
    recap = ChunkExtract(
        action_items=[
            ActionItem.model_validate(
                action(task="Provide the production keys", timestamp="00:01:10")
            )
        ]
    )

    merged = merge_chunks([original, recap])
    assert len(merged.action_items) == 1
    assert merged.action_items[0].timestamp == "00:00:35"


def test_duplicate_keeps_the_higher_confidence() -> None:
    low = ChunkExtract(action_items=[ActionItem.model_validate(action(confidence=0.6))])
    high = ChunkExtract(
        action_items=[ActionItem.model_validate(action(timestamp="00:00:44", confidence=0.95))]
    )
    assert merge_chunks([low, high]).action_items[0].confidence == 0.95


def test_recap_can_supply_a_deadline_the_original_lacked() -> None:
    bare = ChunkExtract(action_items=[ActionItem.model_validate(action(due_phrase=None))])
    recap = ChunkExtract(
        action_items=[ActionItem.model_validate(action(timestamp="00:00:52"))]
    )
    assert merge_chunks([bare, recap]).action_items[0].due_phrase == "by Friday"


def test_same_task_different_owners_stays_two_items() -> None:
    """'Deepa produces the list, Meera owns the comms' is two items."""
    chunk = ChunkExtract(
        action_items=[
            ActionItem.model_validate(action(task="produce the affected list", owner="Deepa")),
            ActionItem.model_validate(
                action(task="produce the affected list comms", owner="Meera", timestamp="00:00:44")
            ),
        ]
    )
    assert len(merge_chunks([chunk]).action_items) == 2


def test_distinct_decisions_are_not_merged() -> None:
    chunk = ChunkExtract(
        decisions=[
            Decision(decision="Move go-live to the 15th", timestamp="00:00:44", made_by="Priya"),
            Decision(decision="Cap exports at 500k rows", timestamp="00:00:52", made_by="Priya"),
        ]
    )
    assert len(merge_chunks([chunk]).decisions) == 2


# --- item-level validation ----------------------------------------------


def test_one_bad_item_does_not_lose_the_good_ones() -> None:
    report = ExtractionReport()
    result = _validate_items(
        {"action_items": [action(), action(timestamp="nonsense"), action(task="")]}, report
    )

    assert len(result.action_items) == 1
    assert report.items_invalid == 2
    assert report.items_returned == 3


def test_non_list_section_is_ignored() -> None:
    report = ExtractionReport()
    assert _validate_items({"action_items": "lots"}, report).action_items == []
    assert any("not a list" in note for note in report.notes)


# --- engine end to end --------------------------------------------------


def test_pipeline_resolves_dates_in_python() -> None:
    client = StubClient({"action_items": [action()]}, {"summary": "A summary."})
    result = extract_meeting(transcript(), client, MEETING_DATE, write_summary=False)

    item = result.extract.action_items[0]
    assert item.due_date == date(2026, 9, 4)  # Friday after Thursday 3 Sept
    assert item.due_date_source is DueDateSource.EXPLICIT


def test_unresolvable_phrase_yields_no_date() -> None:
    client = StubClient({"action_items": [action(due_phrase="at some point")]})
    result = extract_meeting(transcript(), client, MEETING_DATE, write_summary=False)

    item = result.extract.action_items[0]
    assert item.due_date is None
    assert item.due_date_source is DueDateSource.NONE


def test_items_citing_a_missing_timestamp_are_dropped() -> None:
    """The hardest guardrail: an unverifiable citation is not shown as fact."""
    client = StubClient({"action_items": [action(timestamp="09:99:99".replace("99", "30"))]})
    result = extract_meeting(transcript(), client, MEETING_DATE, write_summary=False)

    assert result.extract.action_items == []
    assert result.report.items_uncited == 1
    assert any("not in the transcript" in note for note in result.report.notes)


def test_empty_extraction_is_a_valid_result() -> None:
    """Meetings with no action items exist; inventing some is the failure."""
    client = StubClient({"action_items": [], "decisions": []})
    result = extract_meeting(transcript(), client, MEETING_DATE, write_summary=False)

    assert result.extract.item_count == 0
    assert result.report.items_returned == 0


def test_chunk_failure_does_not_abort_the_run() -> None:
    class HalfBroken:
        name, model = "half", "half-1"

        def __init__(self):
            self.calls = 0

        def complete(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("rate limited")
            return LLMResponse(
                text=json.dumps({"action_items": [action()]}), model="m", provider="p"
            )

    client = HalfBroken()
    result = extract_meeting(
        transcript(), client, MEETING_DATE, max_tokens=15, overlap_tokens=3, write_summary=False
    )

    assert result.report.chunk_failures == 1
    assert any("failed" in note for note in result.report.notes)
    assert result.extract.action_items  # later chunks still contributed


def test_unparseable_json_triggers_one_repair_attempt() -> None:
    client = StubClient("not json at all", {"action_items": [action()]})
    result = extract_meeting(transcript(), client, MEETING_DATE, write_summary=False)

    assert result.report.repairs == 1
    assert len(result.extract.action_items) == 1


def test_report_accounts_for_every_returned_item() -> None:
    client = StubClient(
        {
            "action_items": [
                action(),
                action(timestamp="bad"),
                action(timestamp="00:09:30"),
            ]
        }
    )
    report = extract_meeting(
        transcript(), client, MEETING_DATE, write_summary=False
    ).report

    assert report.items_returned == 3
    assert report.items_invalid == 1
    assert report.items_uncited == 1
    assert report.items_kept == 1


def test_summary_failure_is_not_fatal() -> None:
    class SummaryBreaks:
        name, model = "s", "s-1"

        def __init__(self):
            self.calls = 0

        def complete(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    text=json.dumps({"action_items": [action()]}), model="m", provider="p"
                )
            raise RuntimeError("timeout")

    result = extract_meeting(transcript(), SummaryBreaks(), MEETING_DATE, write_summary=True)

    assert result.extract.summary == ""
    assert result.extract.action_items
