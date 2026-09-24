"""Combine per-chunk extractions into one clean set of items.

Two separate sources of duplication have to be handled:

1. **Window overlap.** Chunks share segments by design, so an item near a
   boundary is genuinely extracted twice, seconds apart.
2. **In-meeting recaps.** A chair who closes with "so to summarise, Vikram
   caps exports by Wednesday..." restates commitments made half an hour
   earlier. These are the same commitment, minutes apart in the transcript,
   and must collapse to one entry at the *earliest* timestamp - the rule
   already written into docs/03_labeling_decisions.md.

Recaps are why matching is on text similarity across the whole meeting
rather than only inside overlap regions.
"""

from __future__ import annotations

import re
from typing import Callable, Sequence, TypeVar

from mat.ingest import parse_timestamp

from .schema import ActionItem, ChunkExtract, Cited, Decision, OpenQuestion, Risk

T = TypeVar("T", bound=Cited)

# Jaccard overlap above which two item texts are treated as the same thing.
# Tuned against the corpus recap: high enough that "cap exports at 500k" and
# "add a row cap to exports" merge, low enough that two different tasks
# owned by the same person do not.
SIMILARITY_THRESHOLD = 0.5

# Words that carry no signal for matching two task descriptions.
STOPWORDS = frozenset(
    """
    a an the and or but if then so to of for on in at by with from into over
    is are was were be been being do does did done have has had will would
    can could should shall may might must i we you he she they it this that
    these those there here what which who whom whose when where why how
    about up out as not no yes ok okay just also than too very
    """.split()
)

WORD_RE = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> frozenset[str]:
    """Content words of `text`, lowercased and stemmed very crudely."""
    words = WORD_RE.findall(text.lower())
    return frozenset(_stem(w) for w in words if w not in STOPWORDS and len(w) > 2)


def _stem(word: str) -> str:
    """Chop the endings that make "caps"/"capping"/"capped" look different."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def similarity(left: str, right: str) -> float:
    """Jaccard overlap of content words. 1.0 identical, 0.0 disjoint."""
    a, b = normalize(left), normalize(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def merge_chunks(chunks: Sequence[ChunkExtract]) -> ChunkExtract:
    """Flatten per-chunk results and remove duplicates within each type."""
    return ChunkExtract(
        action_items=_dedupe(
            [item for chunk in chunks for item in chunk.action_items],
            text_of=lambda i: i.task,
            same_subject=_same_owner,
        ),
        decisions=_dedupe(
            [item for chunk in chunks for item in chunk.decisions],
            text_of=lambda d: d.decision,
        ),
        open_questions=_dedupe(
            [item for chunk in chunks for item in chunk.open_questions],
            text_of=lambda q: q.question,
        ),
        risks=_dedupe(
            [item for chunk in chunks for item in chunk.risks],
            text_of=lambda r: r.risk,
        ),
    )


def _same_owner(left: ActionItem, right: ActionItem) -> bool:
    """Two similar tasks owned by different people are two tasks.

    "Deepa produces the list, Meera owns the comms" is one exchange
    yielding two items; matching on text alone would collapse them.
    """
    return left.owner.casefold() == right.owner.casefold()


def _dedupe(
    items: list[T],
    text_of: Callable[[T], str],
    same_subject: Callable[[T, T], bool] | None = None,
) -> list[T]:
    """Collapse near-duplicates, keeping the earliest mention of each."""
    ordered = sorted(items, key=lambda i: (parse_timestamp(i.timestamp), -i.confidence))
    kept: list[T] = []

    for item in ordered:
        match = _find_match(item, kept, text_of, same_subject)
        if match is None:
            kept.append(item)
            continue

        # Same thing said twice: keep the earliest timestamp (already held,
        # since we walk in time order) but the better confidence and the
        # richer field values.
        kept[kept.index(match)] = _absorb(match, item)

    return kept


def _find_match(
    item: T,
    kept: list[T],
    text_of: Callable[[T], str],
    same_subject: Callable[[T, T], bool] | None,
) -> T | None:
    for candidate in kept:
        if same_subject and not same_subject(candidate, item):
            continue
        if similarity(text_of(candidate), text_of(item)) >= SIMILARITY_THRESHOLD:
            return candidate
    return None


def _absorb(kept: T, duplicate: T) -> T:
    """Fold a later restatement into the entry already held.

    A recap often states a deadline the original mention lacked, so the
    later copy can genuinely add information. Timestamp never moves.
    """
    updates: dict[str, object] = {"confidence": max(kept.confidence, duplicate.confidence)}

    if isinstance(kept, ActionItem) and isinstance(duplicate, ActionItem):
        if kept.due_phrase is None and duplicate.due_phrase is not None:
            updates["due_phrase"] = duplicate.due_phrase
        if kept.owner_source.value == "none" and duplicate.owner_source.value != "none":
            updates["owner"] = duplicate.owner
            updates["owner_source"] = duplicate.owner_source

    if isinstance(kept, Decision) and isinstance(duplicate, Decision):
        agreed = list(dict.fromkeys([*kept.agreed_by, *duplicate.agreed_by]))
        updates["agreed_by"] = agreed

    return kept.model_copy(update=updates)


def drop_uncited(items: Sequence[T], valid_timestamps: frozenset[str]) -> tuple[list[T], list[T]]:
    """Split items into (cited, invented) by whether their line exists.

    An item pointing at a timestamp not present in the transcript cannot be
    verified by a human, so it is not shown as fact. This is the hardest
    guardrail in the pipeline and the reason timestamps are mandatory.
    """
    cited = [item for item in items if item.timestamp in valid_timestamps]
    invented = [item for item in items if item.timestamp not in valid_timestamps]
    return cited, invented


__all__ = [
    "ActionItem",
    "Decision",
    "OpenQuestion",
    "Risk",
    "SIMILARITY_THRESHOLD",
    "drop_uncited",
    "merge_chunks",
    "normalize",
    "similarity",
]
