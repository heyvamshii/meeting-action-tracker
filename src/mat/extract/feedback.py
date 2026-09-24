"""Turn human corrections into few-shot examples for the next extraction.

This is what makes the tool improve as it is used. Three signals, in
decreasing order of usefulness:

- **Deleted items** are measured false positives. A person looked at the
  line and said "that is not an action item". Nothing in a hand-written
  prompt is worth as much as a rejection from the actual domain.
- **Added items** are measured misses.
- **Edited owners and dates** show where the model reads the transcript
  differently from the people who were in the room.

Deliberately *not* fine-tuning. With a handful of corrections per meeting,
few-shot examples are the honest tool; fine-tuning on this volume would
overfit and could not be evaluated against the corpus.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

# Kept small on purpose: examples compete with the transcript for the
# context window, and Groq's free tier prices every token of it.
MAX_EXAMPLES_PER_KIND = 4
MIN_EXAMPLES_TO_BOTHER = 2


@dataclass(frozen=True, slots=True)
class FeedbackExamples:
    rejected: tuple[str, ...] = ()
    added: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return len(self.rejected) + len(self.added) >= MIN_EXAMPLES_TO_BOTHER

    def to_prompt_section(self) -> str:
        """Render as an addition to the system prompt, or '' if too thin."""
        if not self:
            return ""

        blocks = ["", "FEEDBACK FROM THIS TEAM", ""]

        if self.rejected:
            blocks.append(
                "A person reviewed these and said they are NOT items worth "
                "recording. Do not extract things like them:"
            )
            blocks += [f'  - "{text}"' for text in self.rejected]
            blocks.append("")

        if self.added:
            blocks.append(
                "A person had to add these by hand because they were missed. "
                "Look for things like them:"
            )
            blocks += [f'  - "{text}"' for text in self.added]
            blocks.append("")

        return "\n".join(blocks)


def collect_feedback(
    connection: sqlite3.Connection, meeting_id: str | None = None
) -> FeedbackExamples:
    """Gather correction examples, most recent first.

    `meeting_id` excludes a meeting from its own feedback, so re-extracting
    a meeting is not handed the answers to that meeting. Without this the
    prompt would be contaminated and any improvement would be an illusion.
    """
    return FeedbackExamples(
        rejected=_query(connection, "delete", "old_value", meeting_id),
        added=_query(connection, "add", "new_value", meeting_id),
    )


def _query(
    connection: sqlite3.Connection,
    action: str,
    column: str,
    exclude_meeting: str | None,
) -> tuple[str, ...]:
    rows = connection.execute(
        f"""
        SELECT DISTINCT c.{column} AS text
        FROM corrections c
        WHERE c.action = ?
          AND c.{column} IS NOT NULL
          AND TRIM(c.{column}) != ''
          AND (? IS NULL OR c.item_id NOT IN (
                SELECT item_id FROM action_items WHERE meeting_id = ?
              ))
        ORDER BY c.correction_id DESC
        LIMIT ?
        """,  # noqa: S608 - column is chosen from a fixed set above
        (action, exclude_meeting, exclude_meeting, MAX_EXAMPLES_PER_KIND),
    ).fetchall()

    return tuple(row["text"].strip() for row in rows)
