"""Phase 4: persistence, idempotent re-extraction, and carry-forward.

The behaviours that matter here are the ones a user would notice going
wrong: a re-run duplicating their board, a re-run reverting their edits, or
last week's unfinished promise failing to reappear.
"""

from __future__ import annotations

from datetime import date

import pytest

from mat.ingest import parse_transcript_text
from conftest import make_extract
from mat.store import (
    action_items,
    corrections,
    delete_meeting,
    get_meeting,
    list_meetings,
    make_item_id,
    meeting_items,
    open_items_before,
    overdue_items,
    owner_workload,
    save_meeting,
    transcript_segments,
    update_item,
)

TRANSCRIPT = """\
[00:00:04] Priya (PM): Where are we on the gateway?
[00:00:35] Mark (Client): I'll get you the production keys by Friday.
[00:00:44] Priya (PM): We're targeting the 15th now.
"""


# --- ids ----------------------------------------------------------------


def test_item_id_is_stable_across_runs() -> None:
    args = ("m1", "action", "00:00:35", "Provide production keys")
    assert make_item_id(*args) == make_item_id(*args)


def test_item_id_survives_rewording() -> None:
    """Reworded but equivalent text keeps the id, so edits and status
    survive a re-extraction that phrases things differently."""
    left = make_item_id("m1", "action", "00:00:35", "Provide the production keys")
    right = make_item_id("m1", "action", "00:00:35", "provide production key")
    assert left == right


def test_item_id_differs_by_meeting_and_timestamp() -> None:
    base = ("action", "00:00:35", "Provide production keys")
    assert make_item_id("m1", *base) != make_item_id("m2", *base)
    assert make_item_id("m1", "action", "00:00:35", "x") != make_item_id(
        "m1", "action", "00:01:35", "x"
    )


# --- saving -------------------------------------------------------------


def test_save_persists_every_section(db) -> None:
    report = save_meeting(db, make_extract())

    assert report.inserted == 4
    items = meeting_items(db, "client-status-2026-09-03")
    assert len(items["action_items"]) == 1
    assert len(items["decisions"]) == 1
    assert len(items["open_questions"]) == 1
    assert len(items["risks"]) == 1


def test_save_stores_transcript_for_citation_lookup(db) -> None:
    transcript = parse_transcript_text(TRANSCRIPT, meeting_id="client-status-2026-09-03")
    save_meeting(db, make_extract(), transcript=transcript)

    segments = transcript_segments(db, "client-status-2026-09-03")
    assert len(segments) == 3
    assert segments[1]["speaker"] == "Mark (Client)"


def test_resaving_updates_rather_than_duplicates(db) -> None:
    save_meeting(db, make_extract())
    report = save_meeting(db, make_extract(summary="Updated summary."))

    assert report.inserted == 0
    assert report.updated == 4
    assert len(action_items(db)) == 1
    assert get_meeting(db, "client-status-2026-09-03")["summary"] == "Updated summary."


def test_meeting_listing_counts_open_work(db) -> None:
    save_meeting(db, make_extract())
    row = list_meetings(db)[0]

    assert row["action_count"] == 1
    assert row["open_count"] == 1


# --- human edits win ----------------------------------------------------


def test_edit_is_logged_and_marks_the_row(db) -> None:
    save_meeting(db, make_extract())
    item = action_items(db)[0]

    assert update_item(db, "action_items", item["item_id"], "owner", "Mark Chen")

    updated = action_items(db)[0]
    assert updated["owner"] == "Mark Chen"
    assert updated["edited"] == 1

    logged = corrections(db)[0]
    assert logged["field"] == "owner"
    assert logged["old_value"] == "Mark (Client)"
    assert logged["new_value"] == "Mark Chen"


def test_reextraction_does_not_revert_a_human_edit(db) -> None:
    save_meeting(db, make_extract())
    item_id = action_items(db)[0]["item_id"]
    update_item(db, "action_items", item_id, "owner", "Mark Chen")

    report = save_meeting(db, make_extract())

    assert report.preserved == 1
    assert action_items(db)[0]["owner"] == "Mark Chen"


def test_reextraction_does_not_reopen_completed_work(db) -> None:
    """Extraction always reports 'open'; a done item must stay done."""
    save_meeting(db, make_extract())
    item_id = action_items(db)[0]["item_id"]
    db.execute("UPDATE action_items SET status = 'done' WHERE item_id = ?", (item_id,))
    db.commit()

    save_meeting(db, make_extract())
    assert action_items(db)[0]["status"] == "done"


def test_update_rejects_unknown_table_or_field(db) -> None:
    save_meeting(db, make_extract())
    item_id = action_items(db)[0]["item_id"]

    with pytest.raises(ValueError, match="unknown table"):
        update_item(db, "secrets; DROP TABLE meetings", item_id, "owner", "x")
    with pytest.raises(ValueError, match="unknown field"):
        update_item(db, "action_items", item_id, "nonexistent", "x")


def test_update_returns_false_for_a_missing_item(db) -> None:
    assert update_item(db, "action_items", "nope", "owner", "x") is False


# --- carry-forward ------------------------------------------------------


def test_open_items_from_earlier_meetings_resurface(db) -> None:
    save_meeting(db, make_extract())
    save_meeting(db, make_extract(meeting_id="followup-2026-09-10", meeting_date=date(2026, 9, 10)))

    carried = open_items_before(db, date(2026, 9, 10))
    assert len(carried) == 1
    assert carried[0]["meeting_id"] == "client-status-2026-09-03"


def test_completed_items_do_not_resurface(db) -> None:
    save_meeting(db, make_extract())
    db.execute("UPDATE action_items SET status = 'done'")
    db.commit()

    assert open_items_before(db, date(2026, 9, 10)) == []


def test_overdue_uses_the_due_date(db) -> None:
    save_meeting(db, make_extract())

    assert overdue_items(db, date(2026, 9, 4)) == []  # due today, not overdue
    assert len(overdue_items(db, date(2026, 9, 8))) == 1


def test_undated_items_are_never_overdue(db) -> None:
    save_meeting(db, make_extract(due=None))
    assert overdue_items(db, date(2027, 1, 1)) == []


def test_restated_commitment_links_to_the_earlier_one(db) -> None:
    """The same promise made again should read as one ageing item."""
    save_meeting(db, make_extract())
    save_meeting(
        db,
        make_extract(
            meeting_id="followup-2026-09-10",
            meeting_date=date(2026, 9, 10),
            task="Provide the production keys",
        ),
    )

    followup = action_items(db, meeting_id="followup-2026-09-10")[0]
    original = action_items(db, meeting_id="client-status-2026-09-03")[0]
    assert followup["carried_from"] == original["item_id"]


def test_different_owner_is_not_treated_as_the_same_commitment(db) -> None:
    save_meeting(db, make_extract())
    save_meeting(
        db,
        make_extract(
            meeting_id="followup-2026-09-10",
            meeting_date=date(2026, 9, 10),
            owner="Arun (Dev)",
        ),
    )

    assert action_items(db, meeting_id="followup-2026-09-10")[0]["carried_from"] is None


def test_links_only_point_backwards_in_time(db) -> None:
    save_meeting(db, make_extract(meeting_id="later-2026-09-20", meeting_date=date(2026, 9, 20)))
    save_meeting(db, make_extract())

    assert action_items(db, meeting_id="later-2026-09-20")[0]["carried_from"] is None


# --- queries ------------------------------------------------------------


def test_filtering_by_owner_and_status(db) -> None:
    save_meeting(db, make_extract())

    assert len(action_items(db, owner="Mark (Client)")) == 1
    assert action_items(db, owner="Nobody") == []
    assert len(action_items(db, status="open")) == 1
    assert action_items(db, status="done") == []


def test_owner_workload_counts_outstanding_work(db) -> None:
    save_meeting(db, make_extract())
    save_meeting(
        db,
        make_extract(
            meeting_id="second-2026-09-10", meeting_date=date(2026, 9, 10), task="Ship the cap"
        ),
    )
    db.execute("UPDATE action_items SET status = 'done' WHERE task = 'Ship the cap'")
    db.commit()

    row = next(r for r in owner_workload(db) if r["owner"] == "Mark (Client)")
    assert row["total"] == 2
    assert row["done"] == 1
    assert row["outstanding"] == 1


def test_deleting_a_meeting_removes_its_items(db) -> None:
    save_meeting(db, make_extract())

    assert delete_meeting(db, "client-status-2026-09-03") is True
    assert action_items(db) == []
    assert delete_meeting(db, "client-status-2026-09-03") is False
