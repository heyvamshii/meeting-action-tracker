"""Phase 6: the correction loop.

What matters here is that a correction is never lost, never silently
reverted, and never leaks back into the meeting it came from.
"""

from __future__ import annotations

from datetime import date

import pytest

from mat.extract import collect_feedback
from mat.extract.feedback import FeedbackExamples
from mat.store import (
    action_items,
    add_item,
    connect,
    correction_stats,
    corrections,
    delete_item,
    deleted_items,
    list_meetings,
    meeting_items,
    migrate,
    restore_item,
    save_meeting,
    update_item,
)

from conftest import make_extract

MEETING = "client-status-2026-09-03"


@pytest.fixture()
def seeded(db):
    """A database with one stored meeting."""
    save_meeting(db, make_extract())
    return db


def only_action(connection):
    return action_items(connection, meeting_id=MEETING)[0]


# --- migration ----------------------------------------------------------


def test_migration_is_idempotent(seeded) -> None:
    """A fresh schema already has the columns, so nothing is applied twice."""
    assert migrate(seeded) == []


def test_migration_adds_columns_to_an_older_database() -> None:
    """Simulates the database that existed before Phase 6."""
    connection = connect(":memory:")
    connection.execute("ALTER TABLE action_items DROP COLUMN deleted")
    connection.commit()

    assert "action_items.deleted" in migrate(connection)
    assert migrate(connection) == []
    connection.close()


# --- rejecting (false positives) ----------------------------------------


def test_rejected_item_disappears_from_every_view(seeded) -> None:
    item_id = only_action(seeded)["item_id"]
    assert delete_item(seeded, "action_items", item_id) is True

    assert action_items(seeded) == []
    assert meeting_items(seeded, MEETING)["action_items"] == []
    assert list_meetings(seeded)[0]["action_count"] == 0


def test_rejection_keeps_the_evidence(seeded) -> None:
    """A rejected item is a measured false positive; it must survive."""
    item_id = only_action(seeded)["item_id"]
    delete_item(seeded, "action_items", item_id)

    kept = deleted_items(seeded, "action_items", MEETING)
    assert len(kept) == 1
    assert kept[0]["item_id"] == item_id


def test_rejection_is_logged_with_the_text(seeded) -> None:
    row = only_action(seeded)
    delete_item(seeded, "action_items", row["item_id"])

    entry = corrections(seeded)[0]
    assert entry["action"] == "delete"
    assert entry["old_value"] == row["task"]


def test_restore_brings_an_item_back(seeded) -> None:
    item_id = only_action(seeded)["item_id"]
    delete_item(seeded, "action_items", item_id)

    assert restore_item(seeded, "action_items", item_id) is True
    assert len(action_items(seeded)) == 1
    assert [entry["action"] for entry in corrections(seeded)][:2] == ["restore", "delete"]


def test_reextraction_does_not_resurrect_a_rejected_item(seeded) -> None:
    """Re-running extraction must not undo someone's rejection."""
    delete_item(seeded, "action_items", only_action(seeded)["item_id"])
    save_meeting(seeded, make_extract())

    assert action_items(seeded) == []


def test_delete_rejects_an_unknown_table(seeded) -> None:
    with pytest.raises(ValueError, match="unknown table"):
        delete_item(seeded, "meetings", "whatever")


def test_delete_returns_false_for_a_missing_item(seeded) -> None:
    assert delete_item(seeded, "action_items", "nope") is False


# --- adding (misses) ----------------------------------------------------


def test_added_item_is_marked_human_and_certain(seeded) -> None:
    item_id = add_item(
        seeded,
        "action_items",
        MEETING,
        {"task": "Book the pen test", "owner": "Priya (PM)", "timestamp": "00:00:44"},
    )

    added = next(row for row in action_items(seeded) if row["item_id"] == item_id)
    assert added["origin"] == "human"
    assert added["edited"] == 1
    assert added["confidence"] == 1.0


def test_added_item_survives_reextraction(seeded) -> None:
    add_item(seeded, "action_items", MEETING, {"task": "Book the pen test"})
    save_meeting(seeded, make_extract())

    assert any(row["task"] == "Book the pen test" for row in action_items(seeded))


def test_adding_is_logged_as_a_miss(seeded) -> None:
    add_item(seeded, "action_items", MEETING, {"task": "Book the pen test"})
    entry = corrections(seeded)[0]

    assert entry["action"] == "add"
    assert entry["new_value"] == "Book the pen test"


def test_added_item_requires_text(seeded) -> None:
    with pytest.raises(ValueError, match="required"):
        add_item(seeded, "action_items", MEETING, {"task": "   "})


def test_add_rejects_an_unknown_table(seeded) -> None:
    with pytest.raises(ValueError, match="unknown table"):
        add_item(seeded, "nonsense", MEETING, {"task": "x"})


# --- statistics ---------------------------------------------------------


def test_stats_separate_the_three_signals(seeded) -> None:
    row = only_action(seeded)
    update_item(seeded, "action_items", row["item_id"], "owner", "Mark Chen")
    add_item(seeded, "action_items", MEETING, {"task": "Book the pen test"})
    delete_item(seeded, "action_items", row["item_id"])

    stats = correction_stats(seeded)
    assert stats["edits"] == 1
    assert stats["adds"] == 1
    assert stats["deletes"] == 1
    assert stats["model_items"] == 4  # the human-added one is not a model item


# --- few-shot examples --------------------------------------------------


def test_no_examples_until_there_is_enough_signal(seeded) -> None:
    assert not collect_feedback(seeded)
    assert collect_feedback(seeded).to_prompt_section() == ""


def test_rejections_and_additions_become_examples(seeded) -> None:
    delete_item(seeded, "action_items", only_action(seeded)["item_id"])
    add_item(seeded, "action_items", MEETING, {"task": "Book the pen test"})

    examples = collect_feedback(seeded)
    assert examples
    section = examples.to_prompt_section()
    assert "NOT items worth recording" in section
    assert "Book the pen test" in section


def test_a_meeting_is_never_shown_its_own_corrections(seeded) -> None:
    """Otherwise re-extracting hands the model the answers and any
    improvement is an illusion."""
    delete_item(seeded, "action_items", only_action(seeded)["item_id"])
    add_item(seeded, "action_items", MEETING, {"task": "Book the pen test"})

    assert not collect_feedback(seeded, meeting_id=MEETING)


def test_examples_from_other_meetings_are_used(seeded) -> None:
    save_meeting(
        seeded, make_extract(meeting_id="other-2026-09-10", meeting_date=date(2026, 9, 10))
    )
    other = action_items(seeded, meeting_id="other-2026-09-10")[0]
    delete_item(seeded, "action_items", other["item_id"])
    add_item(seeded, "action_items", "other-2026-09-10", {"task": "Book the pen test"})

    assert collect_feedback(seeded, meeting_id=MEETING)


def test_example_count_is_capped(seeded) -> None:
    for index in range(10):
        add_item(seeded, "action_items", MEETING, {"task": f"Task number {index}"})

    assert len(collect_feedback(seeded).added) <= 4


def test_empty_examples_render_nothing() -> None:
    assert FeedbackExamples().to_prompt_section() == ""
