"""The extraction pipeline: transcript in, validated MeetingExtract out.

    chunk -> LLM per chunk -> validate per item -> merge/dedupe
          -> drop uncited -> resolve dates -> summarise

Every stage is allowed to discard, never to invent. Items are validated one
at a time rather than as a batch so that a single malformed entry costs one
item instead of a whole chunk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable

from pydantic import ValidationError

from mat.config import settings
from mat.ingest import Transcript
from mat.llm import JSONParseError, parse_json_object

from .chunker import Chunk, chunk_transcript
from .dates import resolve_due_phrase
from .feedback import FeedbackExamples
from .merge import drop_uncited, merge_chunks
from .prompts import CHUNK_PROMPT, REPAIR_PROMPT, SUMMARY_PROMPT, SYSTEM_PROMPT
from .schema import (
    ActionItem,
    ChunkExtract,
    Decision,
    MeetingExtract,
    OpenQuestion,
    Risk,
)

MAX_TOKENS = settings.llm_max_tokens
MAX_REPAIR_ATTEMPTS = 1

SECTIONS: tuple[tuple[str, type], ...] = (
    ("action_items", ActionItem),
    ("decisions", Decision),
    ("open_questions", OpenQuestion),
    ("risks", Risk),
)

ProgressCallback = Callable[[int, int], None]


@dataclass
class ExtractionReport:
    """What happened, including everything that was thrown away.

    Discards are counted, not hidden. A pipeline that quietly drops a third
    of the model's output while reporting success is worse than one that
    fails loudly.
    """

    chunks: int = 0
    llm_calls: int = 0
    repairs: int = 0
    chunk_failures: int = 0
    items_returned: int = 0
    items_invalid: int = 0
    items_uncited: int = 0
    items_duplicate: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def items_kept(self) -> int:
        return (
            self.items_returned
            - self.items_invalid
            - self.items_uncited
            - self.items_duplicate
        )

    def summary_line(self) -> str:
        return (
            f"{self.items_kept} kept from {self.items_returned} returned "
            f"({self.items_invalid} invalid, {self.items_uncited} uncited, "
            f"{self.items_duplicate} duplicate) over {self.chunks} chunks, "
            f"{self.llm_calls} LLM calls"
        )


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    extract: MeetingExtract
    report: ExtractionReport


def extract_meeting(
    transcript: Transcript,
    client,
    meeting_date: date,
    title: str = "",
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
    write_summary: bool = True,
    on_progress: ProgressCallback | None = None,
    feedback: "FeedbackExamples | None" = None,
) -> ExtractionResult:
    """Run the full pipeline over one transcript.

    `feedback` carries few-shot examples built from earlier human
    corrections (see feedback.py). It is optional so the pipeline stays
    reproducible: Phase 7 measures with it off, then on.
    """
    report = ExtractionReport()
    system_prompt = SYSTEM_PROMPT
    if feedback:
        system_prompt += feedback.to_prompt_section()
        report.notes.append(
            f"using {len(feedback.rejected)} rejected and {len(feedback.added)} "
            "added examples from earlier corrections"
        )
    chunks = chunk_transcript(transcript, max_tokens, overlap_tokens)
    report.chunks = len(chunks)

    participants = [s for s in transcript.speakers if s != "UNKNOWN"]
    per_chunk: list[ChunkExtract] = []

    for chunk in chunks:
        if on_progress:
            on_progress(chunk.index, len(chunks))
        per_chunk.append(
            _extract_chunk(
                client=client,
                chunk=chunk,
                total=len(chunks),
                title=title or transcript.meeting_id,
                meeting_date=meeting_date,
                participants=participants,
                report=report,
                system_prompt=system_prompt,
            )
        )

    # Unverifiable items are dropped before dedupe, not after: an invented
    # citation that happens to resemble a real item would otherwise be
    # counted as a duplicate and vanish from the accounting.
    valid_timestamps = frozenset(segment.timestamp for segment in transcript)
    per_chunk = [_drop_uncited_chunk(chunk, valid_timestamps, report) for chunk in per_chunk]

    before = _count(per_chunk)
    merged = merge_chunks(per_chunk)
    report.items_duplicate = before - _count([merged])

    sections = {name: list(getattr(merged, name)) for name, _ in SECTIONS}
    sections["action_items"] = [
        _resolve_dates(item, meeting_date) for item in sections["action_items"]
    ]

    extract = MeetingExtract(
        meeting_id=transcript.meeting_id,
        meeting_date=meeting_date,
        title=title,
        participants=participants,
        summary="",
        **sections,
    )

    if write_summary and extract.item_count:
        extract = extract.model_copy(
            update={"summary": _summarise(client, extract, report)}
        )

    return ExtractionResult(extract=extract, report=report)


# --- per chunk ----------------------------------------------------------


def _extract_chunk(
    client,
    chunk: Chunk,
    total: int,
    title: str,
    meeting_date: date,
    participants: list[str],
    report: ExtractionReport,
    system_prompt: str = SYSTEM_PROMPT,
) -> ChunkExtract:
    prompt = CHUNK_PROMPT.format(
        title=title,
        meeting_date=meeting_date.isoformat(),
        participants=", ".join(participants) or "unknown",
        index=chunk.index + 1,
        total=total,
        start=chunk.start_timestamp,
        end=chunk.end_timestamp,
        transcript=chunk.to_text(),
    )

    try:
        payload = _complete_json(client, prompt, report, system_prompt)
    except Exception as exc:  # noqa: BLE001 - one chunk must not kill the run
        report.chunk_failures += 1
        report.notes.append(f"chunk {chunk.index + 1}/{total} failed: {exc}")
        return ChunkExtract()

    return _validate_items(payload, report)


def _complete_json(
    client, prompt: str, report: ExtractionReport, system_prompt: str = SYSTEM_PROMPT
) -> dict:
    """One call, with a single repair attempt if the JSON is unusable."""
    response = client.complete(
        prompt, system=system_prompt, temperature=0.0, max_tokens=MAX_TOKENS, json_mode=True
    )
    report.llm_calls += 1

    try:
        return parse_json_object(response.text)
    except JSONParseError as exc:
        if MAX_REPAIR_ATTEMPTS < 1:
            raise
        report.repairs += 1
        repair = client.complete(
            f"{prompt}\n\n{REPAIR_PROMPT.format(error=exc)}",
            system=SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=MAX_TOKENS,
            json_mode=True,
        )
        report.llm_calls += 1
        return parse_json_object(repair.text)


def _validate_items(payload: dict, report: ExtractionReport) -> ChunkExtract:
    """Validate item by item so one bad entry costs one item, not a chunk."""
    sections: dict[str, list] = {}

    for name, model in SECTIONS:
        raw = payload.get(name) or []
        if not isinstance(raw, list):
            report.notes.append(f"'{name}' was not a list; ignored")
            raw = []

        kept = []
        for entry in raw:
            report.items_returned += 1
            if not isinstance(entry, dict):
                report.items_invalid += 1
                continue
            try:
                kept.append(model.model_validate(entry))
            except ValidationError as exc:
                report.items_invalid += 1
                report.notes.append(f"invalid {name[:-1]}: {_first_error(exc)}")
        sections[name] = kept

    return ChunkExtract(**sections)


def _first_error(exc: ValidationError) -> str:
    error = exc.errors()[0]
    location = ".".join(str(part) for part in error["loc"]) or "item"
    return f"{location}: {error['msg']}"


# --- post processing ----------------------------------------------------


def _drop_uncited_chunk(
    chunk: ChunkExtract, valid_timestamps: frozenset[str], report: ExtractionReport
) -> ChunkExtract:
    """Remove items whose cited line does not exist in the transcript."""
    sections: dict[str, list] = {}

    for name, _ in SECTIONS:
        cited, invented = drop_uncited(getattr(chunk, name), valid_timestamps)
        report.items_uncited += len(invented)
        for item in invented:
            report.notes.append(
                f"dropped {name[:-1]} citing {item.timestamp}, which is not in the transcript"
            )
        sections[name] = cited

    return ChunkExtract(**sections)


def _resolve_dates(item: ActionItem, meeting_date: date) -> ActionItem:
    """Turn the spoken phrase into a date, in Python. See dates.py."""
    due_date, source = resolve_due_phrase(item.due_phrase, meeting_date)
    return item.model_copy(update={"due_date": due_date, "due_date_source": source})


def _summarise(client, extract: MeetingExtract, report: ExtractionReport) -> str:
    """Summarise from the extracted items, not the raw transcript.

    Summarising the items keeps the prose consistent with the board: the
    summary cannot mention a commitment that is not also an action item.
    """
    lines = [
        *(f"DECISION: {d.decision}" for d in extract.decisions),
        *(f"ACTION: {a.task} ({a.owner})" for a in extract.action_items),
        *(f"OPEN: {q.question}" for q in extract.open_questions),
        *(f"RISK: {r.risk}" for r in extract.risks),
    ]

    prompt = SUMMARY_PROMPT.format(
        title=extract.title or extract.meeting_id,
        meeting_date=extract.meeting_date.isoformat(),
        participants=", ".join(extract.participants) or "unknown",
        items="\n".join(lines),
    )

    try:
        payload = _complete_json(client, prompt, report)
    except Exception as exc:  # noqa: BLE001 - a missing summary is not fatal
        report.notes.append(f"summary failed: {exc}")
        return ""

    summary = payload.get("summary", "")
    return summary.strip() if isinstance(summary, str) else ""


def _count(chunks: list[ChunkExtract]) -> int:
    return sum(
        len(getattr(chunk, name)) for chunk in chunks for name, _ in SECTIONS
    )
