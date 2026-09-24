"""Assign speakers to transcript segments.

Three paths, tried in order of trustworthiness:

1. **Already labeled.** A Zoom or Teams export carries speakers. Believe it
   and do nothing - guessing over real labels can only lose information.
2. **LLM attribution** against a closed roster. Constrained to the supplied
   participants, so it can decline but cannot invent a person.
3. **Heuristic attribution.** Hand-offs ("Karthik, you're up") and
   self-introductions ("Ravi here") name a single speaker each. High
   precision, low coverage - it resolves the lines it can prove and
   abstains everywhere else.

Whatever cannot be resolved stays UNKNOWN. A wrong owner is worse than a
missing one, because a wrong owner looks confidently correct on a board.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from mat.config import settings

from .models import Transcript, TranscriptSegment, UNKNOWN_SPEAKER
from .roster import Roster

# A following line this tightly coupled in time is the same person still
# talking. Wider than this and we abstain rather than guess.
CONTINUATION_GAP_SECONDS = 1.0

# Smaller batches attribute better than large ones: less to keep straight
# per call, and reasoning models spend a large share of their output budget
# before emitting the first token of JSON.
SEGMENTS_PER_LLM_BATCH = 20
CONTEXT_SEGMENTS = 3

# Reasoning models spend a large share of the output budget before the
# first token of JSON - gpt-oss-120b used 102 completion tokens to produce
# `{"ok":true}` - so this cannot be sized for the JSON alone. It is capped
# from config because on Groq's free tier max_tokens counts toward the
# 8000-tokens-per-minute request limit.
LLM_MAX_TOKENS = settings.llm_max_tokens

# "Karthik, you're up" / "over to you, Deepa" / "Aditi." as a whole line.
HANDOFF_PATTERNS = (
    r"(?:over to you|back to you)[,\s]+(?P<name>[A-Z][a-z]+)",
    r"(?P<name>[A-Z][a-z]+)[,\s]+(?:you'?re up|go ahead|your turn|start us off|kick us off)",
    r"(?:let'?s hear from|what about|how about)\s+(?P<name>[A-Z][a-z]+)",
    r"(?P<name>[A-Z][a-z]+)[,\s]+(?:what|how|can you|do you|anything)\b",
)

SELF_ID_PATTERNS = (
    r"^(?:this is|it'?s)\s+(?P<name>[A-Z][a-z]+)\b",
    r"^(?P<name>[A-Z][a-z]+)\s+(?:here|speaking)\b",
)

# A line that is just a name is a hand-off: "Karthik." "Aditi?"
BARE_NAME_RE = re.compile(r"^(?P<name>[A-Z][a-z]+)\s*[.?!,]?$")

_HANDOFF_RES = tuple(re.compile(p, re.IGNORECASE) for p in HANDOFF_PATTERNS)
_SELF_ID_RES = tuple(re.compile(p, re.IGNORECASE) for p in SELF_ID_PATTERNS)


@dataclass(frozen=True, slots=True)
class AttributionResult:
    transcript: Transcript
    method: str  # already-labeled | llm | heuristic | none
    assigned: int
    unknown: int
    notes: tuple[str, ...] = ()

    @property
    def coverage(self) -> float:
        total = self.assigned + self.unknown
        return self.assigned / total if total else 0.0


def is_labeled(transcript: Transcript) -> bool:
    """True when every segment already names a speaker."""
    return bool(transcript) and all(s.speaker != UNKNOWN_SPEAKER for s in transcript)


def attribute_speakers(
    transcript: Transcript,
    participants: list[str] | tuple[str, ...] | None = None,
    client=None,
) -> AttributionResult:
    """Fill in speakers, preferring the most trustworthy source available."""
    if is_labeled(transcript):
        return AttributionResult(
            transcript=transcript,
            method="already-labeled",
            assigned=len(transcript),
            unknown=0,
            notes=("transcript arrived with speakers; left untouched",),
        )

    roster = Roster(participants) if participants else Roster.from_transcript(transcript)
    if not roster:
        return _result(
            transcript,
            "none",
            ("no participants supplied and none found in the transcript; "
             "speakers left as UNKNOWN",),
        )

    if client is not None:
        try:
            labeled = _attribute_with_llm(transcript, roster, client)
            return _result(labeled, "llm", (f"roster: {', '.join(roster.labels)}",))
        except Exception as exc:  # noqa: BLE001 - any provider failure falls back
            fallback = _attribute_heuristically(transcript, roster)
            return _result(
                fallback,
                "heuristic",
                (f"LLM attribution failed ({exc}); fell back to heuristics",),
            )

    return _result(
        _attribute_heuristically(transcript, roster),
        "heuristic",
        ("no LLM client supplied; used hand-off and self-introduction cues only",),
    )


def _result(transcript: Transcript, method: str, notes: tuple[str, ...]) -> AttributionResult:
    assigned = sum(1 for s in transcript if s.speaker != UNKNOWN_SPEAKER)
    return AttributionResult(
        transcript=transcript,
        method=method,
        assigned=assigned,
        unknown=len(transcript) - assigned,
        notes=notes,
    )


# --- heuristic path -----------------------------------------------------


def _attribute_heuristically(transcript: Transcript, roster: Roster) -> Transcript:
    """Assign only where there is direct evidence. Everything else abstains.

    An earlier version carried each name forward until the next cue. It
    scored 16% precision on the corpus, because real meetings alternate
    speakers far faster than they name each other. Naming only the segment
    the evidence actually points at trades coverage for precision, which is
    the right way round: a wrong owner looks confidently correct on a board,
    a missing one visibly needs a human.

    Continuation is extended by one segment only for audio-sourced
    transcripts, where end times are measured. Text transcripts carry no
    end time - the parser synthesises one from the next line's start - so
    a zero gap there means nothing at all and must not be read as evidence.
    """
    evidence = _collect_evidence(transcript, roster)
    if not evidence:
        return transcript

    segments = list(transcript.segments)
    timings_are_real = transcript.source == "audio"

    for index, speaker in evidence.items():
        segments[index] = segments[index].with_speaker(speaker)

        following = index + 1
        if (
            timings_are_real
            and following < len(segments)
            and following not in evidence
            and _is_continuation(segments, following)
        ):
            segments[following] = segments[following].with_speaker(speaker)

    return transcript.with_segments(segments)


def _is_continuation(segments: list[TranscriptSegment], index: int) -> bool:
    """True when a segment follows the previous one with no real pause."""
    gap = segments[index].start - segments[index - 1].end
    return 0 <= gap < CONTINUATION_GAP_SECONDS


def _collect_evidence(transcript: Transcript, roster: Roster) -> dict[int, str]:
    """Map segment index -> speaker, only where there is direct evidence."""
    evidence: dict[int, str] = {}

    for index, segment in enumerate(transcript.segments):
        text = segment.text.strip()

        for pattern in _SELF_ID_RES:
            match = pattern.search(text)
            if match and (label := roster.match(match.group("name"))):
                evidence[index] = label
                break

        target = _handoff_target(text, roster)
        if target and index + 1 < len(transcript):
            # The next speaker is named, not this one.
            evidence.setdefault(index + 1, target)

    return evidence


def _handoff_target(text: str, roster: Roster) -> str | None:
    """The person handed to, if this line hands over."""
    # Only the tail of a line hands over: "...that's it. Karthik, you're up."
    tail = re.split(r"(?<=[.?!])\s+", text)[-1]

    # A trailing sentence that is only a name is a hand-off: "Good. Karthik."
    for candidate in (text, tail):
        bare = BARE_NAME_RE.match(candidate)
        if bare and (label := roster.match(bare.group("name"))):
            return label

    for pattern in _HANDOFF_RES:
        match = pattern.search(tail)
        if match and (label := roster.match(match.group("name"))):
            return label

    return None


# --- LLM path -----------------------------------------------------------

SYSTEM_PROMPT = """\
You label meeting transcript lines with the speaker who said them.

Rules:
- Use ONLY names from the participant list. Never invent a name.
- If you cannot tell who spoke a line, answer "UNKNOWN". A wrong name is
  worse than UNKNOWN.
- Consecutive lines are usually the same speaker until someone is
  addressed, asked a question, or hands over.
- A line that addresses someone by name is spoken by a different person.
Reply with JSON only."""


def _attribute_with_llm(transcript: Transcript, roster: Roster, client) -> Transcript:
    segments = list(transcript.segments)

    for start in range(0, len(segments), SEGMENTS_PER_LLM_BATCH):
        batch = segments[start : start + SEGMENTS_PER_LLM_BATCH]
        context = segments[max(0, start - CONTEXT_SEGMENTS) : start]

        assignments = _request_batch(client, roster, batch, context, offset=start)
        for index, label in assignments.items():
            segments[index] = segments[index].with_speaker(label)

    return transcript.with_segments(segments)


def _request_batch(client, roster: Roster, batch, context, offset: int) -> dict[int, str]:
    from mat.llm import parse_json_object

    lines = [
        f"{offset + i}: {segment.text}"
        for i, segment in enumerate(batch)
    ]
    context_lines = [f"({s.speaker}) {s.text}" for s in context]

    prompt = "\n".join(
        [
            f"Participants: {', '.join(roster.labels)}",
            "",
            *(["Preceding lines for context:", *context_lines, ""] if context_lines else []),
            "Label each numbered line below with its speaker.",
            'Respond as {"assignments": [{"line": <number>, "speaker": "<name>"}]}',
            "",
            *lines,
        ]
    )

    response = client.complete(
        prompt,
        system=SYSTEM_PROMPT,
        temperature=0.0,
        max_tokens=LLM_MAX_TOKENS,
        json_mode=True,
    )
    payload = parse_json_object(response.text)

    valid_indices = range(offset, offset + len(batch))
    assignments: dict[int, str] = {}

    for entry in payload.get("assignments", payload.get("items", [])):
        if not isinstance(entry, dict):
            continue
        try:
            index = int(entry.get("line"))
        except (TypeError, ValueError):
            continue
        if index not in valid_indices:
            continue
        # Anything outside the roster is dropped, not trusted.
        if label := roster.match(str(entry.get("speaker", ""))):
            assignments[index] = label

    return assignments
