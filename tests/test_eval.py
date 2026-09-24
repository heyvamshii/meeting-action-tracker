"""Phase 7: the scorer.

A scorer that flatters the system is worse than no scorer, so these tests
are mostly about the ways it could cheat.
"""

from __future__ import annotations

from datetime import date


from mat.eval import aggregate, score_meeting
from mat.extract import ActionItem, Decision, MeetingExtract


def extract(*items, decisions=()) -> MeetingExtract:
    return MeetingExtract(
        meeting_id="m",
        meeting_date=date(2026, 9, 3),
        action_items=list(items),
        decisions=list(decisions),
    )


def action(task, owner="Mark (Client)", timestamp="00:00:35", due=date(2026, 9, 4)):
    return ActionItem(
        task=task,
        owner=owner,
        owner_source="speaker",
        due_phrase="by Friday" if due else None,
        due_date=due,
        due_date_source="explicit" if due else "none",
        timestamp=timestamp,
        confidence=0.9,
    )


def gold(task, owner="Mark (Client)", timestamp="00:00:35", due="2026-09-04"):
    return {"task": task, "owner": owner, "timestamp": timestamp, "due_date": due}


def labels(*action_items, decisions=()) -> dict:
    return {"action_items": list(action_items), "decisions": list(decisions)}


# --- matching -----------------------------------------------------------


def test_reworded_item_still_matches() -> None:
    score = score_meeting(
        extract(action("Provide the production keys")),
        labels(gold("Provide production payment gateway keys")),
    )
    assert score.actions.matched == 1
    assert score.actions.f1 == 1.0


def test_unrelated_item_does_not_match() -> None:
    score = score_meeting(
        extract(action("Write the regression suite")),
        labels(gold("Provide production keys")),
    )
    assert score.actions.matched == 0
    assert score.actions.precision == 0.0


def test_same_wording_far_apart_in_time_does_not_match() -> None:
    """The recap case: identical text twenty minutes later is a different
    mention, and must not be credited as the same item."""
    score = score_meeting(
        extract(action("Provide the production keys", timestamp="00:40:00")),
        labels(gold("Provide the production keys", timestamp="00:00:35")),
    )
    assert score.actions.matched == 0


def test_duplicate_predictions_cannot_earn_two_matches() -> None:
    """Otherwise failing to deduplicate would improve the score."""
    score = score_meeting(
        extract(
            action("Provide the production keys"),
            action("Provide production keys", timestamp="00:00:44"),
        ),
        labels(gold("Provide the production keys")),
    )
    assert score.actions.matched == 1
    assert score.actions.false_positives == 1
    assert score.actions.precision == 0.5


def test_matching_is_one_to_one_across_gold_items() -> None:
    score = score_meeting(
        extract(action("Provide the production keys")),
        labels(
            gold("Provide the production keys"),
            gold("Provide the production keys again", timestamp="00:00:44"),
        ),
    )
    assert score.actions.matched == 1
    assert score.actions.misses == 1


# --- metrics ------------------------------------------------------------


def test_empty_prediction_scores_zero_not_an_error() -> None:
    score = score_meeting(extract(), labels(gold("Provide keys")))
    assert score.actions.precision == 0.0
    assert score.actions.recall == 0.0
    assert score.actions.f1 == 0.0


def test_empty_gold_and_empty_prediction_is_not_a_failure() -> None:
    """A meeting with no action items is a real case, not a bug."""
    score = score_meeting(extract(), labels())
    assert score.actions.matched == 0
    assert score.actions.false_positives == 0
    assert score.actions.misses == 0


def test_false_positives_on_an_empty_gold_set() -> None:
    score = score_meeting(extract(action("Invented task")), labels())
    assert score.actions.precision == 0.0
    assert score.actions.false_positives == 1


def test_misses_and_spurious_items_are_named() -> None:
    score = score_meeting(
        extract(action("Write the regression suite")),
        labels(gold("Provide production keys")),
    )
    assert any("Provide production keys" in text for text in score.missed_items)
    assert any("regression suite" in text for text in score.spurious_items)


# --- field accuracy -----------------------------------------------------


def test_owner_compared_on_first_name_only() -> None:
    score = score_meeting(
        extract(action("Provide the production keys", owner="Mark")),
        labels(gold("Provide the production keys", owner="Mark (Client)")),
    )
    assert score.owner_accuracy == 1.0


def test_wrong_owner_is_counted() -> None:
    score = score_meeting(
        extract(action("Provide the production keys", owner="Arun (Dev)")),
        labels(gold("Provide the production keys", owner="Mark (Client)")),
    )
    assert score.owner_accuracy == 0.0


def test_date_accuracy_requires_an_exact_date() -> None:
    score = score_meeting(
        extract(action("Provide the production keys", due=date(2026, 9, 7))),
        labels(gold("Provide the production keys", due="2026-09-04")),
    )
    assert score.date_accuracy == 0.0


def test_missing_date_matches_a_missing_gold_date() -> None:
    score = score_meeting(
        extract(action("Provide the production keys", due=None)),
        labels(gold("Provide the production keys", due=None)),
    )
    assert score.date_accuracy == 1.0


def test_field_accuracy_ignores_unmatched_items() -> None:
    """Finding the wrong things and getting details wrong are different
    failures and must not be conflated."""
    score = score_meeting(
        extract(action("Something else entirely", owner="Nobody")),
        labels(gold("Provide production keys")),
    )
    assert score.owner_compared == 0


# --- guardrail ----------------------------------------------------------


def test_unsupported_citations_are_counted() -> None:
    score = score_meeting(
        extract(action("Provide the production keys", timestamp="00:09:59")),
        labels(gold("Provide the production keys", timestamp="00:09:59")),
        valid_timestamps=frozenset({"00:00:35"}),
    )
    assert score.unsupported == 1


# --- aggregation --------------------------------------------------------


def test_totals_are_pooled_not_averaged() -> None:
    """A six-item meeting must not outvote a sixty-item one."""
    small = score_meeting(extract(action("Provide the production keys")),
                          labels(gold("Provide the production keys")))
    large = score_meeting(
        extract(*[action(f"Task {i}", timestamp="00:00:35") for i in range(9)]),
        labels(*[gold(f"Task {i}", timestamp="00:00:35") for i in range(9)]),
    )
    totals = aggregate([small, large])
    assert totals["action_items"].predicted == 10
    assert totals["action_items"].gold == 10


def test_sections_are_scored_independently() -> None:
    score = score_meeting(
        extract(decisions=[Decision(decision="Move go-live to the 15th",
                                    made_by="Priya", timestamp="00:00:44")]),
        labels(decisions=[{"decision": "Go-live moved to the 15th",
                           "timestamp": "00:00:44"}]),
    )
    assert score.sections["decisions"].matched == 1
    assert score.sections["action_items"].predicted == 0
