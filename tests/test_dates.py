"""Phase 3: deadline resolution.

Date arithmetic lives in Python precisely so it can be tested like this.
MEETING_DATE is Thursday 3 September 2026 throughout.
"""

from __future__ import annotations

from datetime import date

import pytest

from mat.extract import DueDateSource, resolve_due_phrase

MEETING_DATE = date(2026, 9, 3)  # a Thursday


def resolve(phrase):
    return resolve_due_phrase(phrase, MEETING_DATE)


# --- explicit -----------------------------------------------------------


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("by Friday", date(2026, 9, 4)),
        ("Monday", date(2026, 9, 7)),
        ("by next Monday", date(2026, 9, 14)),
        ("today", MEETING_DATE),
        ("end of day", MEETING_DATE),
        ("tomorrow", date(2026, 9, 4)),
        ("before the 12th", date(2026, 9, 12)),
        ("by the 1st", date(2026, 10, 1)),
        ("December the first", date(2026, 12, 1)),
        ("December 1", date(2026, 12, 1)),
        ("the 15th of September", date(2026, 9, 15)),
        ("2026-11-30", date(2026, 11, 30)),
    ],
)
def test_explicit_phrases(phrase, expected) -> None:
    resolved, source = resolve(phrase)
    assert resolved == expected
    assert source is DueDateSource.EXPLICIT


def test_same_weekday_means_next_week() -> None:
    """'Thursday' said on a Thursday means the next one, not today."""
    resolved, _ = resolve("by Thursday")
    assert resolved == date(2026, 9, 10)


def test_day_already_past_rolls_to_next_month() -> None:
    resolved, _ = resolve("the 1st")
    assert resolved == date(2026, 10, 1)


def test_month_already_past_rolls_to_next_year() -> None:
    resolved, _ = resolve("January 15")
    assert resolved == date(2027, 1, 15)


# --- inferred -----------------------------------------------------------


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("this week", date(2026, 9, 6)),  # Sunday of the meeting week
        ("next week", date(2026, 9, 13)),
        ("end of the week", date(2026, 9, 6)),
        ("end of next week", date(2026, 9, 13)),
        ("end of the month", date(2026, 9, 30)),
        ("end of the quarter", date(2026, 9, 30)),
        ("end of next sprint", date(2026, 9, 17)),
        ("within 3 days", date(2026, 9, 6)),
        ("in a week", date(2026, 9, 10)),
    ],
)
def test_inferred_phrases(phrase, expected) -> None:
    resolved, source = resolve(phrase)
    assert resolved == expected
    assert source is DueDateSource.INFERRED


# --- none ---------------------------------------------------------------


@pytest.mark.parametrize(
    "phrase",
    [None, "", "   ", "soon", "at some point", "eventually", "before GA", "when it's ready"],
)
def test_unresolvable_phrases_invent_nothing(phrase) -> None:
    """An unrecognised phrase must not become a plausible-looking date."""
    resolved, source = resolve(phrase)
    assert resolved is None
    assert source is DueDateSource.NONE


# --- properties ---------------------------------------------------------


@pytest.mark.parametrize(
    "phrase",
    ["by Friday", "this week", "next week", "the 12th", "end of the month", "tomorrow"],
)
def test_resolved_dates_are_never_in_the_past(phrase) -> None:
    resolved, _ = resolve(phrase)
    assert resolved is not None and resolved >= MEETING_DATE


def test_explicit_beats_inferred_when_both_appear() -> None:
    """'by Friday this week' names a day; the day wins."""
    resolved, source = resolve("by Friday this week")
    assert source is DueDateSource.EXPLICIT
    assert resolved == date(2026, 9, 4)
