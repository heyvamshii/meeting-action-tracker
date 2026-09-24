"""Cross-meeting tracker: what is outstanding, who owes it, what has slipped.

This is the tab that makes the tool a tracker rather than a note taker. It
reads across every stored meeting, so a promise made three weeks ago and
never closed is at the top of the page rather than buried in an old file.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st

from mat.store import action_items, open_items_before, overdue_items, owner_workload

from ..components import VERIFY, empty_state, needs_verification, verification_legend
from ..exports import action_items_csv, jira_csv

STATUSES = ["open", "in_progress", "done", "dropped"]


def render(connection) -> None:
    st.subheader("Tracker")

    rows = action_items(connection)
    if not rows:
        empty_state(
            "No action items stored yet.",
            "Extract a meeting on the Upload tab, or run "
            "`python scripts/track.py load data/extracts/<file>.json`.",
        )
        return

    as_of = st.date_input(
        "As of",
        value=date.today(),
        help="Items due before this date count as overdue.",
    )

    _overdue_section(connection, as_of)
    _outstanding_section(connection, as_of)
    _filters_and_table(connection, rows)
    _workload_chart(connection)
    _exports(rows)


def _overdue_section(connection, as_of: date) -> None:
    overdue = overdue_items(connection, as_of)
    if not overdue:
        return

    st.error(f"{len(overdue)} overdue item(s)")
    for row in overdue:
        flag = f" {VERIFY}" if needs_verification(row) else ""
        st.markdown(
            f"**{row['due_date']}** · {row['owner']}{flag} — {row['task']}  \n"
            f"<span style='color:gray;font-size:0.85em'>"
            f"{row['title'] or row['meeting_id']} · {row['timestamp']}</span>",
            unsafe_allow_html=True,
        )


def _outstanding_section(connection, as_of: date) -> None:
    carried = open_items_before(connection, as_of)
    overdue_ids = {row["item_id"] for row in overdue_items(connection, as_of)}
    remaining = [row for row in carried if row["item_id"] not in overdue_ids]

    if not remaining:
        return

    with st.expander(f"Open, not yet overdue ({len(remaining)})", expanded=False):
        for row in remaining:
            carried_marker = " · carried forward" if row["carried_from"] else ""
            due = row["due_date"] or "no date"
            st.markdown(f"**{due}** · {row['owner']} — {row['task']}{carried_marker}")


def _filters_and_table(connection, rows) -> None:
    st.markdown("### All action items")

    owners = sorted({row["owner"] for row in rows})
    left, right = st.columns(2)
    owner = left.selectbox("Owner", ["All", *owners])
    status = right.selectbox("Status", ["All", *STATUSES])

    filtered = action_items(
        connection,
        owner=None if owner == "All" else owner,
        status=None if status == "All" else status,
    )

    if not filtered:
        empty_state("No items match those filters.")
        return

    frame = pd.DataFrame(
        [
            {
                "": VERIFY if needs_verification(row) else "",
                "Date": row["meeting_date"],
                "Owner": row["owner"],
                "Task": row["task"],
                "Due": row["due_date"] or "—",
                "Status": row["status"],
                "Conf": round(row["confidence"], 2),
                "Meeting": row["title"] or row["meeting_id"],
            }
            for row in filtered
        ]
    )
    st.dataframe(frame, use_container_width=True, hide_index=True)
    verification_legend()


def _workload_chart(connection) -> None:
    workload = owner_workload(connection)
    if not workload:
        return

    st.markdown("### Commitments per person")
    frame = pd.DataFrame(
        [
            {
                "Owner": row["owner"],
                "Done": row["done"] or 0,
                "Outstanding": row["outstanding"] or 0,
            }
            for row in workload
        ]
    ).melt(id_vars="Owner", var_name="State", value_name="Items")

    figure = px.bar(
        frame,
        x="Items",
        y="Owner",
        color="State",
        orientation="h",
        color_discrete_map={"Done": "#4c9a6a", "Outstanding": "#c98a2e"},
        height=max(220, 42 * frame["Owner"].nunique()),
    )
    figure.update_layout(
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        legend_title_text="",
        xaxis_title="",
        yaxis_title="",
    )
    st.plotly_chart(figure, use_container_width=True)


def _exports(rows) -> None:
    st.markdown("### Export")
    left, right = st.columns(2)

    left.download_button(
        "Download CSV",
        data=action_items_csv(rows),
        file_name="action-items.csv",
        mime="text/csv",
        help="Full export, including how each owner and date was determined.",
        use_container_width=True,
    )
    right.download_button(
        "Download Jira CSV",
        data=jira_csv(rows),
        file_name="action-items-jira.csv",
        mime="text/csv",
        help="Jira import format. Provenance caveats are folded into the description.",
        use_container_width=True,
    )
