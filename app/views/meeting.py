"""One meeting: summary, decisions, action items, questions, risks.

Everything is editable in place. An extracted item is a proposal, not a
fact, and the fastest way to make the tool trustworthy is to make the
correction cheaper than the complaint. Edits are logged and survive
re-extraction (see docs/09_storage_notes.md).
"""

from __future__ import annotations

from datetime import date

import streamlit as st

from mat.extract import ItemStatus, QuestionStatus
from mat.store import (
    add_item,
    delete_item,
    get_meeting,
    list_meetings,
    meeting_items,
    update_item,
)

from ..components import (
    VERIFY,
    empty_state,
    confidence_caption,
    needs_verification,
    timestamp_button,
    verification_legend,
)
from ..exports import meeting_markdown


def render(connection) -> None:
    meetings = list_meetings(connection)
    if not meetings:
        empty_state(
            "No meetings stored yet.",
            "Use the Upload tab to extract one.",
        )
        return

    # The id is included in the label because uploads default to the file
    # name and often have no title - without it two meetings from the same
    # day are indistinguishable in the picker.
    labels = {
        f"{row['meeting_date']} · {row['title'] or row['meeting_id']}"
        + (f"  ({row['action_count']} actions)" if row["title"] else
           f"  ({row['action_count']} actions)"): row["meeting_id"]
        for row in meetings
    }
    default = st.session_state.get("selected_meeting")
    keys = list(labels)
    index = next(
        (i for i, key in enumerate(keys) if labels[key] == default),
        0,
    )

    choice = st.selectbox(
        "Meeting", keys, index=index, help="Newest first. Uploads select themselves here."
    )
    meeting_id = labels[choice]
    st.session_state["selected_meeting"] = meeting_id

    meeting = get_meeting(connection, meeting_id)
    items = meeting_items(connection, meeting_id)

    if not any(items[section] for section in items):
        st.error(
            f"**{meeting_id}** stored no items. The transcript was processed but "
            "nothing could be extracted from it — open the Transcript tab and check "
            "whether it reads as sentences or as fragments."
        )

    _header(meeting, items)
    verification_legend()

    tabs = st.tabs(
        [
            f"Actions ({len(items['action_items'])})",
            f"Decisions ({len(items['decisions'])})",
            f"Questions ({len(items['open_questions'])})",
            f"Risks ({len(items['risks'])})",
        ]
    )
    with tabs[0]:
        _action_items(connection, items["action_items"], meeting_id)
        _add_action_form(connection, meeting_id)
    with tabs[1]:
        _decisions(connection, items["decisions"], meeting_id)
    with tabs[2]:
        _questions(connection, items["open_questions"], meeting_id)
    with tabs[3]:
        _risks(connection, items["risks"], meeting_id)


def _header(meeting, items) -> None:
    st.subheader(meeting["title"] or meeting["meeting_id"])

    actions = items["action_items"]
    flagged = sum(1 for row in actions if needs_verification(row))
    edited = sum(1 for row in actions if row["edited"])

    columns = st.columns(4)
    columns[0].metric("Date", str(meeting["meeting_date"]))
    columns[1].metric("Action items", len(actions))
    columns[2].metric("Need checking", flagged)
    columns[3].metric("Edited by hand", edited)

    if meeting["summary"]:
        st.info(meeting["summary"])

    st.download_button(
        "Download as Markdown",
        data=meeting_markdown(meeting, items),
        file_name=f"{meeting['meeting_id']}.md",
        mime="text/markdown",
    )


def _action_items(connection, rows, meeting_id: str) -> None:
    if not rows:
        empty_state("No action items in this meeting.")
        return

    for row in rows:
        flag = f"{VERIFY} " if needs_verification(row) else ""
        due = f" · due {row['due_date']}" if row["due_date"] else ""
        edited = " · edited" if row["edited"] else ""
        title = f"{flag}{row['owner']} — {row['task'][:70]}{due}{edited}"

        with st.expander(title, expanded=False):
            _action_editor(connection, row, meeting_id)


def _action_editor(connection, row, meeting_id: str) -> None:
    st.caption(confidence_caption(row))
    if row["due_phrase"]:
        st.caption(f'heard: "{row["due_phrase"]}"')
    timestamp_button(row, key=f"ts_action_{row['item_id']}")

    task = st.text_area("Task", value=row["task"], key=f"task_{row['item_id']}")
    left, middle, right = st.columns(3)
    owner = left.text_input("Owner", value=row["owner"], key=f"owner_{row['item_id']}")
    status = middle.selectbox(
        "Status",
        [s.value for s in ItemStatus],
        index=[s.value for s in ItemStatus].index(row["status"]),
        key=f"status_{row['item_id']}",
    )
    due = right.date_input(
        "Due date",
        value=date.fromisoformat(row["due_date"]) if row["due_date"] else None,
        key=f"due_{row['item_id']}",
    )

    save, reject = st.columns([1, 1])
    if save.button("Save", key=f"save_action_{row['item_id']}", type="primary"):
        changes = {
            "task": task,
            "owner": owner,
            "status": status,
            "due_date": due.isoformat() if due else None,
        }
        _apply(connection, "action_items", row, changes)

    if reject.button(
        "Not an action item",
        key=f"del_action_{row['item_id']}",
        help="Hides it and records a false positive. Restorable from the Feedback tab.",
    ):
        delete_item(connection, "action_items", row["item_id"])
        st.rerun()


def _add_action_form(connection, meeting_id: str) -> None:
    """Let a person record something the model missed.

    Added items are the measured misses, and the counterpart to rejections.
    Without this the feedback loop only ever sees false positives.
    """
    with st.expander("Add an action item the extractor missed"):
        with st.form(f"add_action_{meeting_id}", clear_on_submit=True):
            task = st.text_input("Task")
            left, middle, right = st.columns(3)
            owner = left.text_input("Owner", value="UNASSIGNED")
            timestamp = middle.text_input("Transcript time", value="00:00:00")
            due = right.date_input("Due date", value=None)

            if st.form_submit_button("Add", type="primary"):
                if not task.strip():
                    st.warning("A task is required.")
                    return
                add_item(
                    connection,
                    "action_items",
                    meeting_id,
                    {
                        "task": task.strip(),
                        "owner": owner.strip() or "UNASSIGNED",
                        "owner_source": "explicit" if owner.strip() else "none",
                        "due_date": due.isoformat() if due else None,
                        "due_date_source": "explicit" if due else "none",
                        "due_phrase": None,
                        "status": "open",
                        "timestamp": timestamp.strip() or "00:00:00",
                    },
                )
                st.success("Added.")
                st.rerun()


def _decisions(connection, rows, meeting_id: str) -> None:
    if not rows:
        empty_state("No decisions in this meeting.")
        return

    for row in rows:
        with st.expander(f"{row['decision'][:80]} — {row['made_by']}"):
            st.caption(confidence_caption(row))
            if row["agreed_by"]:
                st.caption(f"agreed by: {row['agreed_by']}")
            timestamp_button(row, key=f"ts_decision_{row['item_id']}")

            text = st.text_area("Decision", value=row["decision"], key=f"d_{row['item_id']}")
            made_by = st.text_input("Made by", value=row["made_by"], key=f"mb_{row['item_id']}")
            if st.button("Save", key=f"save_decision_{row['item_id']}", type="primary"):
                _apply(connection, "decisions", row, {"decision": text, "made_by": made_by})


def _questions(connection, rows, meeting_id: str) -> None:
    if not rows:
        empty_state("No open questions in this meeting.")
        return

    for row in rows:
        with st.expander(f"{row['question'][:80]} ({row['status']})"):
            st.caption(confidence_caption(row))
            timestamp_button(row, key=f"ts_question_{row['item_id']}")

            text = st.text_area("Question", value=row["question"], key=f"q_{row['item_id']}")
            statuses = [s.value for s in QuestionStatus]
            status = st.selectbox(
                "Status",
                statuses,
                index=statuses.index(row["status"]),
                key=f"qs_{row['item_id']}",
            )
            if st.button("Save", key=f"save_question_{row['item_id']}", type="primary"):
                _apply(connection, "open_questions", row, {"question": text, "status": status})


def _risks(connection, rows, meeting_id: str) -> None:
    if not rows:
        empty_state("No risks recorded in this meeting.")
        return

    for row in rows:
        with st.expander(f"({row['severity']}) {row['risk'][:80]}"):
            st.caption(confidence_caption(row))
            timestamp_button(row, key=f"ts_risk_{row['item_id']}")

            text = st.text_area("Risk", value=row["risk"], key=f"r_{row['item_id']}")
            severities = ["low", "medium", "high"]
            severity = st.selectbox(
                "Severity",
                severities,
                index=severities.index(row["severity"]),
                key=f"rs_{row['item_id']}",
            )
            if st.button("Save", key=f"save_risk_{row['item_id']}", type="primary"):
                _apply(connection, "risks", row, {"risk": text, "severity": severity})


def _apply(connection, table: str, row, changes: dict) -> None:
    """Write only the fields that actually changed.

    Writing everything would log a correction for each untouched field and
    drown the real signal in Phase 6's audit trail.
    """
    applied = 0
    for field, value in changes.items():
        if str(row[field]) == str(value):
            continue
        update_item(connection, table, row["item_id"], field, value)
        applied += 1

    if applied:
        st.success(f"Saved {applied} change(s).")
        st.rerun()
    else:
        st.info("Nothing changed.")
