"""Score an extraction against hand-labelled ground truth.

The matching rule is the whole design problem. Extracted text never equals
labelled text word for word, so an exact comparison would score zero and a
loose one would score anything. A predicted item matches a gold item when
**both**:

- their texts overlap by at least `MATCH_SIMILARITY` (Jaccard over stemmed
  content words), and
- their timestamps are within `MATCH_WINDOW_SECONDS` of each other.

The time window is what stops "write the policy" matching a different
"write the policy" said twenty minutes later, which is exactly the recap
case the Phase 4 dedupe defect lives in.

Matching is greedy and one-to-one: each gold item is claimed at most once,
so two predictions of the same commitment score as one match and one false
positive rather than two matches. Without that, failing to dedupe would
*improve* the score.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from mat.extract import MeetingExtract
from mat.extract.merge import similarity
from mat.ingest import parse_timestamp

MATCH_SIMILARITY = 0.4
MATCH_WINDOW_SECONDS = 180

SECTIONS: tuple[tuple[str, str], ...] = (
    ("action_items", "task"),
    ("decisions", "decision"),
    ("open_questions", "question"),
    ("risks", "risk"),
)


@dataclass(frozen=True, slots=True)
class SectionScore:
    section: str
    predicted: int
    gold: int
    matched: int

    @property
    def precision(self) -> float:
        return self.matched / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.matched / self.gold if self.gold else 0.0

    @property
    def f1(self) -> float:
        if not self.precision or not self.recall:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)

    @property
    def false_positives(self) -> int:
        return self.predicted - self.matched

    @property
    def misses(self) -> int:
        return self.gold - self.matched


@dataclass
class MeetingScore:
    meeting_id: str
    sections: dict[str, SectionScore] = field(default_factory=dict)

    owner_correct: int = 0
    owner_compared: int = 0
    date_correct: int = 0
    date_compared: int = 0
    unsupported: int = 0  # items citing a line that is not in the transcript

    missed_items: list[str] = field(default_factory=list)
    spurious_items: list[str] = field(default_factory=list)

    @property
    def owner_accuracy(self) -> float:
        return self.owner_correct / self.owner_compared if self.owner_compared else 0.0

    @property
    def date_accuracy(self) -> float:
        return self.date_correct / self.date_compared if self.date_compared else 0.0

    @property
    def actions(self) -> SectionScore:
        return self.sections["action_items"]


def load_ground_truth(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def score_meeting(
    extract: MeetingExtract,
    gold: dict,
    valid_timestamps: frozenset[str] | None = None,
) -> MeetingScore:
    """Compare one extraction against one set of hand labels."""
    score = MeetingScore(meeting_id=extract.meeting_id)

    for section, text_field in SECTIONS:
        predicted = list(getattr(extract, section))
        gold_items = list(gold.get(section, []))

        pairs = _match(predicted, gold_items, text_field)
        score.sections[section] = SectionScore(
            section=section,
            predicted=len(predicted),
            gold=len(gold_items),
            matched=len(pairs),
        )

        matched_gold = {id(gold_item) for _, gold_item in pairs}
        matched_pred = {id(item) for item, _ in pairs}

        score.missed_items += [
            f"{section}: {item[text_field]}"
            for item in gold_items
            if id(item) not in matched_gold
        ]
        score.spurious_items += [
            f"{section}: {getattr(item, text_field)}"
            for item in predicted
            if id(item) not in matched_pred
        ]

        if section == "action_items":
            _score_fields(pairs, score)

    if valid_timestamps is not None:
        score.unsupported = sum(
            1 for item in extract.all_cited() if item.timestamp not in valid_timestamps
        )

    return score


def _match(predicted: list, gold_items: list[dict], text_field: str) -> list[tuple]:
    """Greedy one-to-one matching, best pairs first.

    One-to-one matters: it makes a duplicated prediction cost a false
    positive instead of earning a second match.
    """
    candidates = []
    for item in predicted:
        for gold_item in gold_items:
            score = similarity(getattr(item, text_field), gold_item[text_field])
            if score < MATCH_SIMILARITY:
                continue
            gap = abs(
                parse_timestamp(item.timestamp) - parse_timestamp(gold_item["timestamp"])
            )
            if gap > MATCH_WINDOW_SECONDS:
                continue
            candidates.append((score, -gap, item, gold_item))

    candidates.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)

    used_pred: set[int] = set()
    used_gold: set[int] = set()
    pairs = []

    for _, _, item, gold_item in candidates:
        if id(item) in used_pred or id(gold_item) in used_gold:
            continue
        used_pred.add(id(item))
        used_gold.add(id(gold_item))
        pairs.append((item, gold_item))

    return pairs


def _score_fields(pairs: list[tuple], score: MeetingScore) -> None:
    """Owner and due-date accuracy, over matched action items only.

    Scoring these over unmatched items would conflate two different
    failures: finding the wrong things, and getting the details wrong on
    the right things.
    """
    for item, gold_item in pairs:
        gold_owner = str(gold_item.get("owner", "UNASSIGNED"))
        score.owner_compared += 1
        if _same_person(item.owner, gold_owner):
            score.owner_correct += 1

        gold_date = gold_item.get("due_date")
        score.date_compared += 1
        predicted_date = item.due_date.isoformat() if item.due_date else None
        if predicted_date == gold_date:
            score.date_correct += 1


def _same_person(left: str, right: str) -> bool:
    """Compare on first name, so "Mark" and "Mark (Client)" agree."""
    return _first_name(left) == _first_name(right)


def _first_name(label: str) -> str:
    bare = label.split("(")[0].strip().casefold()
    return bare.split()[0] if bare else ""


def aggregate(scores: list[MeetingScore]) -> dict[str, SectionScore]:
    """Corpus-level totals, pooled rather than averaged.

    Averaging per-meeting rates would let a six-item meeting outvote a
    sixty-item one.
    """
    totals: dict[str, SectionScore] = {}

    for section, _ in SECTIONS:
        predicted = sum(s.sections[section].predicted for s in scores if section in s.sections)
        gold = sum(s.sections[section].gold for s in scores if section in s.sections)
        matched = sum(s.sections[section].matched for s in scores if section in s.sections)
        totals[section] = SectionScore(section, predicted, gold, matched)

    return totals
