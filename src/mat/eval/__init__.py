"""Evaluation: extraction output scored against hand-labelled ground truth."""

from .scorer import (
    MATCH_SIMILARITY,
    MATCH_WINDOW_SECONDS,
    MeetingScore,
    SectionScore,
    aggregate,
    load_ground_truth,
    score_meeting,
)

__all__ = [
    "MATCH_SIMILARITY",
    "MATCH_WINDOW_SECONDS",
    "MeetingScore",
    "SectionScore",
    "aggregate",
    "load_ground_truth",
    "score_meeting",
]
