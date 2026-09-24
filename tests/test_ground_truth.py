"""Phase 0 guard: ground truth files must match docs/01_schema.md.

A malformed baseline silently corrupts every Phase 7 metric, so the
labels are validated the same way model output will be.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pytest

from mat.config import GROUND_TRUTH_DIR

TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")
DUE_DATE_SOURCES = {"explicit", "inferred", "none"}
STATUSES = {"open", "in_progress", "done", "dropped"}
QUESTION_STATUSES = {"open", "parked", "answered"}
SEVERITIES = {"low", "medium", "high"}


def _label_files() -> list[Path]:
    return sorted(p for p in GROUND_TRUTH_DIR.glob("*.json") if not p.name.startswith("_"))


def test_corpus_is_not_empty() -> None:
    assert _label_files(), "no labeled meetings yet - Phase 0 is incomplete"


@pytest.mark.parametrize("path", _label_files(), ids=lambda p: p.stem)
def test_label_file_matches_schema(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["meeting_id"] == path.stem
    meeting_date = date.fromisoformat(data["meeting_date"])
    assert data["participants"], "participants must be listed"
    assert data["summary"].strip(), "summary must not be empty"

    for decision in data["decisions"]:
        assert decision["decision"].strip()
        assert decision["made_by"].strip()
        assert TIMESTAMP_RE.match(decision["timestamp"])

    for item in data["action_items"]:
        assert item["task"].strip()
        assert item["owner"].strip()
        assert item["due_date_source"] in DUE_DATE_SOURCES
        assert item["status"] in STATUSES
        assert TIMESTAMP_RE.match(item["timestamp"])

        # The core guardrail: a date and its provenance must agree.
        if item["due_date"] is None:
            assert item["due_date_source"] == "none"
        else:
            assert item["due_date_source"] != "none"
            assert date.fromisoformat(item["due_date"]) >= meeting_date

    for question in data["open_questions"]:
        assert question["question"].strip()
        assert question["status"] in QUESTION_STATUSES
        assert TIMESTAMP_RE.match(question["timestamp"])

    for risk in data["risks"]:
        assert risk["risk"].strip()
        assert risk["severity"] in SEVERITIES


@pytest.mark.parametrize("path", _label_files(), ids=lambda p: p.stem)
def test_every_timestamp_exists_in_transcript(path: Path) -> None:
    """No item may cite a timestamp the transcript does not contain."""
    from mat.config import TRANSCRIPT_DIR

    transcript = TRANSCRIPT_DIR / f"{path.stem}.txt"
    if not transcript.exists():
        pytest.skip(f"transcript not present for {path.stem}")

    present = set(re.findall(r"\[(\d{2}:\d{2}:\d{2})\]", transcript.read_text(encoding="utf-8")))
    data = json.loads(path.read_text(encoding="utf-8"))

    for section in ("decisions", "action_items", "open_questions", "risks"):
        for item in data[section]:
            assert item["timestamp"] in present, (
                f"{section} item cites {item['timestamp']}, absent from transcript"
            )
