"""Turn a spoken deadline into an absolute date, in Python not in the model.

LLMs are unreliable at date arithmetic - ask one what "Friday" means when
the meeting was Thursday 3 September 2026 and it will often answer
confidently and wrongly. So the model is asked only to copy the words it
saw ("by Friday", "this week", "before the 12th"), and the arithmetic
happens here, deterministically and testably.

The returned source is the guardrail:
    explicit  - a specific day was named
    inferred  - a fuzzy period was named and we picked its end
    none      - no deadline was given, and none is invented
"""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta

from .schema import DueDateSource

WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

MONTHS = {name.lower(): number for number, name in enumerate(calendar.month_name) if name}
MONTHS |= {name.lower(): number for number, name in enumerate(calendar.month_abbr) if name}

ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "twentieth": 20, "thirtieth": 30,
}

# "December the first" is as common in speech as "December 1", so the day
# group accepts a spelled-out ordinal as well as digits.
MONTH_ALT = "|".join(sorted(MONTHS, key=len, reverse=True))
ORDINAL_ALT = "|".join(sorted(ORDINAL_WORDS, key=len, reverse=True))
DAY_ALT = rf"\d{{1,2}}(?:st|nd|rd|th)?|{ORDINAL_ALT}"

ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
DAY_ORDINAL_RE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)\b")
MONTH_DAY_RE = re.compile(
    rf"\b(?P<month>{MONTH_ALT})\.?\s+(?:the\s+)?(?P<day>{DAY_ALT})\b",
    re.IGNORECASE,
)
DAY_MONTH_RE = re.compile(
    rf"\b(?:the\s+)?(?P<day>{DAY_ALT})\s+(?:of\s+)?(?P<month>{MONTH_ALT})\b",
    re.IGNORECASE,
)

# Fuzzy periods, resolved to the end of the period they name.
INFERRED_PERIODS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bend of (?:the )?(?:next|following) week\b"), "end_next_week"),
    (re.compile(r"\bend of (?:the )?week\b"), "end_this_week"),
    (re.compile(r"\bend of (?:the )?month\b"), "end_month"),
    (re.compile(r"\bend of (?:the )?quarter\b"), "end_quarter"),
    (re.compile(r"\bend of (?:the )?(?:next |current )?sprint\b"), "end_sprint"),
    (re.compile(r"\bnext (?:week|sprint)\b"), "end_next_week"),
    (re.compile(r"\bthis (?:week|sprint)\b"), "end_this_week"),
    (re.compile(r"\bin (?:a|one) week\b"), "week_out"),
    (re.compile(r"\bin (?:a|one) month\b"), "month_out"),
    (re.compile(r"\bwithin (?:the )?(?:next )?(?P<days>\d+) days?\b"), "n_days"),
)

# A sprint is assumed to be two weeks. Wrong for some teams; visible as a
# named constant rather than a magic number buried in a branch.
SPRINT_DAYS = 14


def resolve_due_phrase(
    phrase: str | None, meeting_date: date
) -> tuple[date | None, DueDateSource]:
    """Resolve a spoken deadline against the meeting date.

    Returns (date, source). Never guesses: anything unrecognised comes back
    as (None, NONE) rather than a plausible-looking date.
    """
    if not phrase or not phrase.strip():
        return None, DueDateSource.NONE

    text = phrase.strip().lower()

    for resolver in (_iso, _relative_day, _month_and_day, _weekday, _day_of_month):
        resolved = resolver(text, meeting_date)
        if resolved is not None:
            return resolved, DueDateSource.EXPLICIT

    period = _period(text, meeting_date)
    if period is not None:
        return period, DueDateSource.INFERRED

    return None, DueDateSource.NONE


# --- explicit -----------------------------------------------------------


def _iso(text: str, _: date) -> date | None:
    match = ISO_RE.search(text)
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def _relative_day(text: str, meeting_date: date) -> date | None:
    if re.search(r"\btoday\b|\bend of (?:the )?day\b|\beod\b", text):
        return meeting_date
    if re.search(r"\btomorrow\b", text):
        return meeting_date + timedelta(days=1)
    return None


def _month_and_day(text: str, meeting_date: date) -> date | None:
    match = MONTH_DAY_RE.search(text) or DAY_MONTH_RE.search(text)
    if not match:
        return None

    month = MONTHS.get(match.group("month").lower())
    day = _day_value(match.group("day"))
    if not month or day is None:
        return None

    for year in (meeting_date.year, meeting_date.year + 1):
        try:
            candidate = date(year, month, day)
        except ValueError:
            return None
        if candidate >= meeting_date:
            return candidate
    return None


def _day_value(token: str) -> int | None:
    """`"12th"` or `"first"` -> 12 / 1."""
    token = token.strip().lower()
    if token in ORDINAL_WORDS:
        return ORDINAL_WORDS[token]
    digits = re.match(r"(\d{1,2})", token)
    return int(digits.group(1)) if digits else None


def _weekday(text: str, meeting_date: date) -> date | None:
    for name, index in WEEKDAYS.items():
        if not re.search(rf"\b{name}\b", text):
            continue
        ahead = (index - meeting_date.weekday()) % 7
        if ahead == 0:
            ahead = 7  # "Friday" said on a Friday means the next one
        if re.search(r"\bnext\b", text):
            ahead += 7
        return meeting_date + timedelta(days=ahead)
    return None


def _day_of_month(text: str, meeting_date: date) -> date | None:
    """"by the 15th" - the next occurrence of that day number."""
    match = DAY_ORDINAL_RE.search(text)
    day = int(match.group(1)) if match else _ordinal_word(text)
    if day is None or not 1 <= day <= 31:
        return None

    for offset in (0, 1):
        month = meeting_date.month + offset
        year = meeting_date.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        if day > calendar.monthrange(year, month)[1]:
            continue
        candidate = date(year, month, day)
        if candidate >= meeting_date:
            return candidate
    return None


def _ordinal_word(text: str) -> int | None:
    for word, value in ORDINAL_WORDS.items():
        if re.search(rf"\b{word}\b", text):
            return value
    return None


# --- inferred -----------------------------------------------------------


def _period(text: str, meeting_date: date) -> date | None:
    for pattern, kind in INFERRED_PERIODS:
        match = pattern.search(text)
        if not match:
            continue

        if kind == "end_this_week":
            return _end_of_week(meeting_date)
        if kind == "end_next_week":
            return _end_of_week(meeting_date) + timedelta(days=7)
        if kind == "end_month":
            last = calendar.monthrange(meeting_date.year, meeting_date.month)[1]
            return date(meeting_date.year, meeting_date.month, last)
        if kind == "end_quarter":
            month = ((meeting_date.month - 1) // 3 + 1) * 3
            return date(meeting_date.year, month, calendar.monthrange(meeting_date.year, month)[1])
        if kind == "end_sprint":
            return meeting_date + timedelta(days=SPRINT_DAYS)
        if kind == "week_out":
            return meeting_date + timedelta(days=7)
        if kind == "month_out":
            return meeting_date + timedelta(days=30)
        if kind == "n_days":
            return meeting_date + timedelta(days=int(match.group("days")))

    return None


def _end_of_week(meeting_date: date) -> date:
    """Sunday of the week the meeting falls in."""
    return meeting_date + timedelta(days=6 - meeting_date.weekday())
