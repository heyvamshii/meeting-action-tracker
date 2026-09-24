"""Pydantic implementation of docs/01_schema.md.

The schema is the guardrail, not decoration. Anything the model returns
that cannot be validated here never reaches the database, so the rules
below are the difference between a task board and a plausible fiction.
"""

from __future__ import annotations

import re
from datetime import date
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")

UNASSIGNED = "UNASSIGNED"


class DueDateSource(str, Enum):
    """Where a due date came from. The hallucination guardrail for dates."""

    EXPLICIT = "explicit"  # "by Friday", "before the 12th"
    INFERRED = "inferred"  # "this week", "next sprint"
    NONE = "none"  # no date was given


class OwnerSource(str, Enum):
    """Where an owner came from.

    Added after Phase 2 measured speaker attribution at 63% accuracy: an
    owner inferred from who was talking inherits that error rate, so the
    UI has to be able to tell the two apart and flag the weaker one.
    """

    EXPLICIT = "explicit"  # "Deepa, you'll review it"
    SPEAKER = "speaker"  # "I'll review it" - only as good as attribution
    NONE = "none"  # nobody was named


class ItemStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    DROPPED = "dropped"


class QuestionStatus(str, Enum):
    OPEN = "open"
    PARKED = "parked"
    ANSWERED = "answered"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Base(BaseModel):
    model_config = ConfigDict(extra="ignore", use_enum_values=False)


class Cited(Base):
    """Anything that must point at a real line of the transcript."""

    timestamp: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("timestamp")
    @classmethod
    def _check_timestamp(cls, value: str) -> str:
        value = value.strip()
        if not TIMESTAMP_RE.match(value):
            raise ValueError(f"timestamp must be HH:MM:SS, got {value!r}")
        return value


class Decision(Cited):
    decision: str = Field(min_length=1)
    made_by: str = Field(default=UNASSIGNED)
    agreed_by: list[str] = Field(default_factory=list)


class ActionItem(Cited):
    task: str = Field(min_length=1)
    owner: str = Field(default=UNASSIGNED)
    owner_source: OwnerSource = OwnerSource.NONE
    # The deadline exactly as spoken ("by Friday", "this week"). The model
    # copies words; dates.py does the arithmetic. See dates.py for why.
    due_phrase: str | None = None
    due_date: date | None = None
    due_date_source: DueDateSource = DueDateSource.NONE
    status: ItemStatus = ItemStatus.OPEN

    @model_validator(mode="after")
    def _provenance_must_agree(self) -> ActionItem:
        """A value and its stated origin cannot contradict each other.

        Without this the model can emit a confident date alongside
        `source: none`, which is exactly the failure the field exists to
        make visible.
        """
        if self.due_date is None and self.due_date_source is not DueDateSource.NONE:
            raise ValueError("due_date_source claims a date but due_date is null")
        if self.due_date is not None and self.due_date_source is DueDateSource.NONE:
            raise ValueError("due_date is set but due_date_source is 'none'")

        if self.owner == UNASSIGNED and self.owner_source is not OwnerSource.NONE:
            raise ValueError("owner_source claims an owner but owner is UNASSIGNED")
        if self.owner != UNASSIGNED and self.owner_source is OwnerSource.NONE:
            raise ValueError("owner is set but owner_source is 'none'")
        return self


class OpenQuestion(Cited):
    question: str = Field(min_length=1)
    raised_by: str = Field(default=UNASSIGNED)
    status: QuestionStatus = QuestionStatus.OPEN


class Risk(Cited):
    risk: str = Field(min_length=1)
    severity: Severity = Severity.LOW


class MeetingExtract(Base):
    """Everything pulled out of one meeting."""

    meeting_id: str
    meeting_date: date
    title: str = ""
    participants: list[str] = Field(default_factory=list)
    summary: str = ""
    decisions: list[Decision] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    open_questions: list[OpenQuestion] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)

    @property
    def item_count(self) -> int:
        return (
            len(self.decisions)
            + len(self.action_items)
            + len(self.open_questions)
            + len(self.risks)
        )

    def all_cited(self) -> list[Cited]:
        return [*self.decisions, *self.action_items, *self.open_questions, *self.risks]


class ChunkExtract(Base):
    """What one chunk returns. No meeting-level fields - those are merged
    once, from the whole transcript, rather than guessed per window."""

    decisions: list[Decision] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    open_questions: list[OpenQuestion] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
