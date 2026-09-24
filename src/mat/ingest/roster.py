"""The list of people in a meeting, and how to match a name to one of them.

Attribution is only useful if a person is spelled the same way everywhere:
"Ravi", "ravi" and "Ravi (EM)" must collapse to one owner, or action items
fragment across variants and the per-person view lies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import UNKNOWN_SPEAKER

# "Ravi (EM)" -> bare name "Ravi", role "EM"
LABEL_RE = re.compile(r"^\s*(?P<name>[^(]+?)\s*(?:\((?P<role>[^)]*)\))?\s*$")


def split_label(label: str) -> tuple[str, str | None]:
    """`Ravi (EM)` -> `('Ravi', 'EM')`."""
    match = LABEL_RE.match(label)
    if not match:
        return label.strip(), None
    return match.group("name").strip(), (match.group("role") or None)


@dataclass(frozen=True, slots=True)
class Participant:
    label: str  # canonical, as it appears in the transcript
    name: str  # bare name, used for vocative matching
    role: str | None = None

    @classmethod
    def parse(cls, label: str) -> Participant:
        name, role = split_label(label)
        return cls(label=label.strip(), name=name, role=role)

    @property
    def first_name(self) -> str:
        return self.name.split()[0] if self.name else self.name


class Roster:
    """A closed set of participants. Attribution may not invent names."""

    def __init__(self, labels: list[str] | tuple[str, ...]) -> None:
        seen: dict[str, Participant] = {}
        for label in labels:
            cleaned = label.strip()
            if not cleaned or cleaned == UNKNOWN_SPEAKER:
                continue
            participant = Participant.parse(cleaned)
            seen.setdefault(participant.first_name.casefold(), participant)
        self._by_first_name = seen

    def __len__(self) -> int:
        return len(self._by_first_name)

    def __bool__(self) -> bool:
        return bool(self._by_first_name)

    def __iter__(self):
        return iter(self._by_first_name.values())

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(p.label for p in self._by_first_name.values())

    @property
    def first_names(self) -> tuple[str, ...]:
        return tuple(p.first_name for p in self._by_first_name.values())

    @classmethod
    def from_transcript(cls, transcript) -> Roster:
        """Build a roster from whatever speakers a transcript already has."""
        return cls([s for s in transcript.speakers if s != UNKNOWN_SPEAKER])

    def match(self, candidate: str) -> str | None:
        """Resolve a loose name to a canonical label, or None.

        Matches on first name only, case-insensitively, so "ravi",
        "Ravi Kumar" and "Ravi (EM)" all resolve to the roster entry.
        Never guesses between two people.
        """
        if not candidate:
            return None

        bare, _ = split_label(candidate.strip())
        if not bare or bare.casefold() == UNKNOWN_SPEAKER.casefold():
            return None

        first = bare.split()[0].casefold()
        participant = self._by_first_name.get(first)
        return participant.label if participant else None

    def find_mentions(self, text: str) -> list[str]:
        """Roster members named in `text`, in order of appearance."""
        found: list[tuple[int, str]] = []
        for participant in self._by_first_name.values():
            pattern = re.compile(rf"\b{re.escape(participant.first_name)}\b", re.IGNORECASE)
            match = pattern.search(text)
            if match:
                found.append((match.start(), participant.label))
        return [label for _, label in sorted(found)]
