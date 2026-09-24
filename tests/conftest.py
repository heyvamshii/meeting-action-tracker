"""Shared fixtures.

`make_extract` builds a small but complete MeetingExtract so storage and
feedback tests exercise every section rather than action items alone.
"""

from __future__ import annotations

from datetime import date

import pytest

from mat.extract import ActionItem, Decision, MeetingExtract, OpenQuestion, Risk
from mat.store import connect


@pytest.fixture()
def db():
    connection = connect(":memory:")
    yield connection
    connection.close()


def make_extract(
    meeting_id: str = "client-status-2026-09-03",
    meeting_date: date = date(2026, 9, 3),
    task: str = "Provide production keys",
    owner: str = "Mark (Client)",
    due: date | None = date(2026, 9, 4),
    **overrides,
) -> MeetingExtract:
    return MeetingExtract(
        meeting_id=meeting_id,
        meeting_date=meeting_date,
        title=overrides.get("title", "Weekly client status"),
        participants=["Priya (PM)", "Mark (Client)"],
        summary=overrides.get("summary", "Keys pending."),
        action_items=[
            ActionItem(
                task=task,
                owner=owner,
                owner_source="speaker",
                due_phrase="by Friday" if due else None,
                due_date=due,
                due_date_source="explicit" if due else "none",
                timestamp="00:00:35",
                confidence=0.9,
            )
        ],
        decisions=[
            Decision(
                decision="Target the 15th",
                made_by="Priya (PM)",
                agreed_by=["Mark (Client)"],
                timestamp="00:00:44",
                confidence=0.95,
            )
        ],
        open_questions=[
            OpenQuestion(question="PCI sign-off?", raised_by="Priya (PM)", timestamp="00:00:04")
        ],
        risks=[Risk(risk="Keys are a single point of failure", timestamp="00:00:35")],
    )
