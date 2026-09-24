"""What people corrected, and what it says about the extractor.

The counts here are observed corrections, not measured accuracy. They only
capture what somebody bothered to fix, so they are a lower bound and are
labelled as such. The hand-labelled corpus remains the real measurement.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from mat.extract import collect_feedback
from mat.store import (
    ITEM_TABLES,
    correction_stats,
    corrections,
    deleted_items,
    list_meetings,
    restore_item,
)

from ..components import empty_state

ACTION_LABELS = {
    "edit": "edited",
    "add": "added by hand (a miss)",
    "delete": "rejected (a false positive)",
    "restore": "restored",
}


def render(connection) -> None:
    st.subheader("Feedback")

    stats = correction_stats(connection)
    if not any(stats[key] for key in ("edits", "adds", "deletes")):
        empty_state(
            "No corrections recorded yet.",
            "Edit, add or reject an item on the Meeting tab and it will appear here.",
        )
        return

    _stats(stats)
    _honesty_note(stats)
    _rejected_section(connection)
    _log(connection)
    _prompt_preview(connection)


def _stats(stats: dict[str, int]) -> None:
    columns = st.columns(4)
    columns[0].metric("Model items", stats["model_items"])
    columns[1].metric("Edited", stats["edits"])
    columns[2].metric("Rejected", stats["deletes"], help="False positives someone found")
    columns[3].metric("Added by hand", stats["adds"], help="Misses someone found")


def _honesty_note(stats: dict[str, int]) -> None:
    model_items = stats["model_items"]
    if not model_items:
        return

    rejected_rate = stats["deletes"] / model_items
    st.caption(
        f"{stats['deletes']} of {model_items} extracted items were rejected "
        f"({rejected_rate:.1%}). This is a **lower bound on the false positive "
        "rate**, not a measurement: it counts only what someone reviewed and "
        "bothered to correct. Unreviewed meetings contribute nothing."
    )


def _rejected_section(connection) -> None:
    meetings = list_meetings(connection)
    rows = [
        (meeting, table, row)
        for meeting in meetings
        for table in ITEM_TABLES
        for row in deleted_items(connection, table, meeting["meeting_id"])
    ]
    if not rows:
        return

    with st.expander(f"Rejected items ({len(rows)}) — kept as evidence, not deleted"):
        for meeting, table, row in rows:
            _, text_column = ITEM_TABLES[table]
            left, right = st.columns([5, 1])
            left.markdown(
                f"~~{row[text_column]}~~  \n"
                f"<span style='color:gray;font-size:0.85em'>"
                f"{meeting['title'] or meeting['meeting_id']} · {row['timestamp']}</span>",
                unsafe_allow_html=True,
            )
            if right.button("Restore", key=f"restore_{row['item_id']}"):
                restore_item(connection, table, row["item_id"])
                st.rerun()


def _log(connection) -> None:
    entries = corrections(connection, limit=200)
    if not entries:
        return

    st.markdown("### Correction log")
    frame = pd.DataFrame(
        [
            {
                "When": row["corrected_at"].replace("T", " ").replace("+00:00", ""),
                "What": ACTION_LABELS.get(row["action"], row["action"]),
                "Kind": row["item_kind"].replace("_", " "),
                "Field": row["field"],
                "From": (row["old_value"] or "")[:60],
                "To": (row["new_value"] or "")[:60],
            }
            for row in entries
        ]
    )
    st.dataframe(frame, use_container_width=True, hide_index=True)


def _prompt_preview(connection) -> None:
    """Show exactly what the corrections add to the next extraction."""
    examples = collect_feedback(connection)

    st.markdown("### What the next extraction will be told")
    if not examples:
        st.info(
            "Not enough corrections yet to build few-shot examples. "
            "They start being used once a couple of items have been rejected or added."
        )
        return

    st.code(examples.to_prompt_section(), language="text")
    st.caption(
        "Appended to the extraction system prompt. A meeting is never shown "
        "its own corrections — that would hand it the answers and make any "
        "improvement an illusion."
    )
