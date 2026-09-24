from __future__ import annotations

import csv
import io
from typing import Sequence

FULL_COLUMNS = [
    "meeting_date",
    "meeting",
    "task",
    "owner",
    "owner_source",
    "due_date",
    "due_date_source",
    "due_phrase",
    "status",
    "confidence",
    "timestamp",
    "edited",
    "carried_from",
]

JIRA_COLUMNS = ["Summary", "Assignee", "Due Date", "Status", "Description", "Labels"]


def _write(rows: list[list[str]], header: Sequence[str]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def action_items_csv(items: Sequence) -> bytes:
    rows = [
        [
            row["meeting_date"],
            row["title"] or row["meeting_id"],
            row["task"],
            row["owner"],
            row["owner_source"],
            row["due_date"] or "",
            row["due_date_source"],
            row["due_phrase"] or "",
            row["status"],
            f"{row['confidence']:.2f}",
            row["timestamp"],
            "yes" if row["edited"] else "no",
            row["carried_from"] or "",
        ]
        for row in items
    ]
    return _write(rows, FULL_COLUMNS)


def jira_csv(items: Sequence) -> bytes:
    """Jira-importable rows, with the caveats carried in the description."""
    rows = []
    for row in items:
        caveats = [
            f"Source: {row['title'] or row['meeting_id']} at {row['timestamp']}",
            f"Owner determined: {row['owner_source']}",
            f"Due date determined: {row['due_date_source']}"
            + (f' (heard: "{row["due_phrase"]}")' if row["due_phrase"] else ""),
            f"Extraction confidence: {row['confidence']:.2f}",
        ]
        if row["owner_source"] == "speaker":
            caveats.append(
                "NOTE: owner inferred from who was speaking, not named aloud. Verify."
            )
        if row["due_date_source"] == "inferred":
            caveats.append("NOTE: due date inferred from a vague phrase. Verify.")

        labels = ["meeting-extract"]
        if row["confidence"] < 0.8 or row["owner_source"] == "speaker":
            labels.append("needs-verification")

        rows.append(
            [
                row["task"],
                row["owner"] if row["owner"] != "UNASSIGNED" else "",
                row["due_date"] or "",
                "Done" if row["status"] == "done" else "To Do",
                "\n".join(caveats),
                " ".join(labels),
            ]
        )
    return _write(rows, JIRA_COLUMNS)


def meeting_markdown(meeting, items: dict) -> bytes:
    """A meeting as plain Markdown, for pasting into a wiki or an email."""
    lines = [
        f"# {meeting['title'] or meeting['meeting_id']}",
        f"_{meeting['meeting_date']}_",
        "",
    ]
    if meeting["summary"]:
        lines += [meeting["summary"], ""]

    if items["decisions"]:
        lines.append("## Decisions")
        lines += [
            f"- {row['decision']} — {row['made_by']} `[{row['timestamp']}]`"
            for row in items["decisions"]
        ]
        lines.append("")

    if items["action_items"]:
        lines.append("## Action items")
        for row in items["action_items"]:
            due = f" — due {row['due_date']}" if row["due_date"] else ""
            flag = " ⚠️" if row["owner_source"] == "speaker" else ""
            lines.append(f"- **{row['owner']}**{flag}: {row['task']}{due} `[{row['timestamp']}]`")
        lines.append("")

    if items["open_questions"]:
        lines.append("## Open questions")
        lines += [
            f"- {row['question']} — raised by {row['raised_by']} ({row['status']})"
            for row in items["open_questions"]
        ]
        lines.append("")

    if items["risks"]:
        lines.append("## Risks")
        lines += [f"- ({row['severity']}) {row['risk']}" for row in items["risks"]]
        lines.append("")

    lines.append("⚠️ = owner inferred from the speaker label, not named aloud.")
    return "\n".join(lines).encode("utf-8")
