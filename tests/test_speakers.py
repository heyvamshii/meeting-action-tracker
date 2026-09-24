"""Phase 2: roster matching, heuristic attribution, LLM output validation.

The LLM path is exercised with a stub client. What matters is not that a
model answers well - it is that a bad answer cannot corrupt the transcript.
"""

from __future__ import annotations

import json

import pytest

from mat.ingest import (
    Roster,
    UNKNOWN_SPEAKER,
    attribute_speakers,
    is_labeled,
    parse_transcript_text,
    split_label,
)
from mat.llm.base import LLMResponse

STANDUP = """\
[00:00:04] UNKNOWN: Morning everyone. Let's go round. Deepa, start us off.
[00:00:14] UNKNOWN: Yesterday I finished the retry logic.
[00:00:28] UNKNOWN: Any blockers?
[00:00:30] UNKNOWN: No blockers.
[00:00:33] UNKNOWN: Good. Karthik.
[00:00:37] UNKNOWN: I'm still on the notification templates.
"""

PARTICIPANTS = ["Ravi (EM)", "Deepa (Dev)", "Karthik (Dev)"]


class StubClient:
    """Returns a canned payload; records the prompts it was given."""

    name = "stub"
    model = "stub-1"

    def __init__(self, payload) -> None:
        self._payload = payload
        self.prompts: list[str] = []

    def complete(self, prompt, system=None, temperature=0.0, max_tokens=4096, json_mode=False):
        self.prompts.append(prompt)
        text = self._payload if isinstance(self._payload, str) else json.dumps(self._payload)
        return LLMResponse(text=text, model=self.model, provider=self.name)


# --- roster -------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Ravi (EM)", ("Ravi", "EM")),
        ("Deepa", ("Deepa", None)),
        ("  Mark (Client)  ", ("Mark", "Client")),
    ],
)
def test_split_label(label, expected) -> None:
    assert split_label(label) == expected


@pytest.mark.parametrize("candidate", ["Ravi", "ravi", "RAVI", "Ravi (EM)", "Ravi Kumar"])
def test_roster_matches_name_variants(candidate) -> None:
    assert Roster(PARTICIPANTS).match(candidate) == "Ravi (EM)"


@pytest.mark.parametrize("candidate", ["Nisha", "", "UNKNOWN", "the PM"])
def test_roster_rejects_non_members(candidate) -> None:
    assert Roster(PARTICIPANTS).match(candidate) is None


def test_roster_deduplicates_and_preserves_order() -> None:
    roster = Roster(["Ravi (EM)", "ravi", "Deepa (Dev)"])
    assert roster.labels == ("Ravi (EM)", "Deepa (Dev)")
    assert len(roster) == 2


def test_roster_ignores_unknown_placeholder() -> None:
    assert not Roster([UNKNOWN_SPEAKER, "  "])


def test_find_mentions_returns_order_of_appearance() -> None:
    roster = Roster(PARTICIPANTS)
    assert roster.find_mentions("Karthik, can you sync with Deepa?") == [
        "Karthik (Dev)",
        "Deepa (Dev)",
    ]


# --- already-labeled path -----------------------------------------------


def test_labeled_transcript_is_left_alone() -> None:
    labeled = parse_transcript_text("[00:00:01] Ravi (EM): Hello.\n", meeting_id="m")
    assert is_labeled(labeled)

    result = attribute_speakers(labeled, PARTICIPANTS)
    assert result.method == "already-labeled"
    assert result.coverage == 1.0
    assert result.transcript.segments[0].speaker == "Ravi (EM)"


def test_no_roster_means_no_guessing() -> None:
    transcript = parse_transcript_text(STANDUP, meeting_id="m")
    result = attribute_speakers(transcript, participants=None)

    assert result.method == "none"
    assert result.assigned == 0


# --- heuristic path -----------------------------------------------------


def test_handoff_names_the_next_speaker_not_the_current_one() -> None:
    transcript = parse_transcript_text(STANDUP, meeting_id="m")
    segments = attribute_speakers(transcript, PARTICIPANTS).transcript.segments

    assert segments[0].speaker == UNKNOWN_SPEAKER  # said "Deepa, start us off"
    assert segments[1].speaker == "Deepa (Dev)"
    assert segments[5].speaker == "Karthik (Dev)"  # after the bare "Karthik."


def test_heuristic_abstains_rather_than_guesses() -> None:
    """Unresolvable lines stay UNKNOWN. A wrong owner is worse than none."""
    transcript = parse_transcript_text(STANDUP, meeting_id="m")
    result = attribute_speakers(transcript, PARTICIPANTS)

    assert result.method == "heuristic"
    assert result.unknown > 0
    assert all(
        s.speaker in (*Roster(PARTICIPANTS).labels, UNKNOWN_SPEAKER) for s in result.transcript
    )


def test_self_introduction_names_the_current_speaker() -> None:
    text = "[00:00:01] UNKNOWN: Ravi here, sorry I'm late.\n[00:00:09] UNKNOWN: No problem.\n"
    segments = attribute_speakers(
        parse_transcript_text(text, meeting_id="m"), PARTICIPANTS
    ).transcript.segments

    assert segments[0].speaker == "Ravi (EM)"
    assert segments[1].speaker == UNKNOWN_SPEAKER


def test_no_continuation_bleed_on_text_transcripts() -> None:
    """Text transcripts have synthesised end times, so a zero gap is
    meaningless and must never be read as 'same speaker continued'."""
    transcript = parse_transcript_text(STANDUP, meeting_id="m")
    segments = attribute_speakers(transcript, PARTICIPANTS).transcript.segments

    # Line 2 is Ravi asking "Any blockers?", immediately after Deepa's line.
    assert segments[2].speaker == UNKNOWN_SPEAKER


def test_mention_mid_sentence_is_not_a_handoff() -> None:
    text = (
        "[00:00:01] UNKNOWN: Deepa reviewed it and caught two bugs before release.\n"
        "[00:00:12] UNKNOWN: Good catch.\n"
    )
    segments = attribute_speakers(
        parse_transcript_text(text, meeting_id="m"), PARTICIPANTS
    ).transcript.segments

    assert segments[1].speaker == UNKNOWN_SPEAKER


# --- LLM path validation ------------------------------------------------


def test_llm_assignments_are_applied() -> None:
    client = StubClient(
        {"assignments": [{"line": 0, "speaker": "Ravi"}, {"line": 1, "speaker": "Deepa"}]}
    )
    result = attribute_speakers(parse_transcript_text(STANDUP, meeting_id="m"), PARTICIPANTS, client)

    assert result.method == "llm"
    assert result.transcript.segments[0].speaker == "Ravi (EM)"
    assert result.transcript.segments[1].speaker == "Deepa (Dev)"


def test_llm_prompt_carries_the_roster() -> None:
    client = StubClient({"assignments": []})
    attribute_speakers(parse_transcript_text(STANDUP, meeting_id="m"), PARTICIPANTS, client)

    assert "Ravi (EM)" in client.prompts[0]


def test_invented_speakers_are_discarded() -> None:
    """A name outside the roster is dropped, never written to a segment."""
    client = StubClient({"assignments": [{"line": 0, "speaker": "Nisha"}]})
    result = attribute_speakers(parse_transcript_text(STANDUP, meeting_id="m"), PARTICIPANTS, client)

    assert result.transcript.segments[0].speaker == UNKNOWN_SPEAKER


def test_out_of_range_line_numbers_are_ignored() -> None:
    client = StubClient({"assignments": [{"line": 999, "speaker": "Ravi"}]})
    result = attribute_speakers(parse_transcript_text(STANDUP, meeting_id="m"), PARTICIPANTS, client)

    assert result.assigned == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"assignments": [{"line": "not a number", "speaker": "Ravi"}]},
        {"assignments": ["Ravi"]},
        {"assignments": [{"speaker": "Ravi"}]},
    ],
)
def test_malformed_entries_are_skipped_not_fatal(payload) -> None:
    result = attribute_speakers(
        parse_transcript_text(STANDUP, meeting_id="m"), PARTICIPANTS, StubClient(payload)
    )
    assert result.assigned == 0


def test_fenced_json_is_recovered() -> None:
    client = StubClient('```json\n{"assignments": [{"line": 0, "speaker": "Ravi"}]}\n```')
    result = attribute_speakers(parse_transcript_text(STANDUP, meeting_id="m"), PARTICIPANTS, client)

    assert result.transcript.segments[0].speaker == "Ravi (EM)"


def test_provider_failure_falls_back_to_heuristics() -> None:
    class BrokenClient:
        name = "broken"
        model = "broken-1"

        def complete(self, *args, **kwargs):
            raise RuntimeError("connection reset")

    result = attribute_speakers(
        parse_transcript_text(STANDUP, meeting_id="m"), PARTICIPANTS, BrokenClient()
    )

    assert result.method == "heuristic"
    assert result.transcript.segments[1].speaker == "Deepa (Dev)"
    assert any("fell back" in note for note in result.notes)
