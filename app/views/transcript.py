"""The transcript, with the lines that produced items marked.

This is the verification surface. Every extracted item cites a timestamp;
this tab is where a person goes to check that the line actually says what
the item claims. Without it the citations are decoration.
"""

from __future__ import annotations

import streamlit as st

from mat.store import list_meetings, meeting_items, transcript_segments

from ..components import empty_state

CITED_STYLE = "background-color:rgba(201,138,46,0.18);border-left:3px solid #c98a2e;"
JUMP_STYLE = "background-color:rgba(76,154,106,0.28);border-left:3px solid #4c9a6a;"


def render(connection) -> None:
    meetings = list_meetings(connection)
    if not meetings:
        empty_state("No meetings stored yet.")
        return

    labels = {
        f"{row['meeting_date']} — {row['title'] or row['meeting_id']}": row["meeting_id"]
        for row in meetings
    }
    keys = list(labels)

    # A timestamp click on another tab selects both the meeting and the line.
    preselected = st.session_state.get("jump_meeting") or st.session_state.get(
        "selected_meeting"
    )
    index = next((i for i, key in enumerate(keys) if labels[key] == preselected), 0)

    choice = st.selectbox("Meeting", keys, index=index, key="transcript_meeting")
    meeting_id = labels[choice]

    segments = transcript_segments(connection, meeting_id)
    if not segments:
        empty_state(
            "No transcript stored for this meeting.",
            "Transcripts are saved when a meeting is extracted through the Upload tab. "
            "Meetings loaded from a JSON extract carry items but no transcript.",
        )
        return

    cited = _cited_timestamps(connection, meeting_id)
    jump_to = st.session_state.get("jump_to")

    left, right = st.columns([3, 1])
    left.caption(
        f"{len(segments)} lines · {len(cited)} produced an extracted item (highlighted)"
    )
    if jump_to and right.button("Clear highlight"):
        st.session_state.pop("jump_to", None)
        st.session_state.pop("jump_meeting", None)
        st.rerun()

    only_cited = st.checkbox("Show only lines that produced items", value=False)

    if jump_to:
        st.success(f"Jumped to {jump_to}")

    rendered = [
        _line_html(row, cited, jump_to)
        for row in segments
        if not only_cited or row["timestamp"] in cited
    ]

    if not rendered:
        empty_state("No lines produced items in this meeting.")
        return

    st.markdown(
        "<div style='font-family:ui-monospace,monospace;font-size:0.86rem;line-height:1.6'>"
        + "".join(rendered)
        + "</div>",
        unsafe_allow_html=True,
    )


def _cited_timestamps(connection, meeting_id: str) -> dict[str, list[str]]:
    """timestamp -> the kinds of item that cite it."""
    items = meeting_items(connection, meeting_id)
    kinds = {
        "action_items": "action",
        "decisions": "decision",
        "open_questions": "question",
        "risks": "risk",
    }

    cited: dict[str, list[str]] = {}
    for table, label in kinds.items():
        for row in items[table]:
            cited.setdefault(row["timestamp"], []).append(label)
    return cited


def _line_html(row, cited: dict[str, list[str]], jump_to: str | None) -> str:
    timestamp = row["timestamp"]
    style = "padding:2px 8px;"

    if timestamp == jump_to:
        style += JUMP_STYLE
    elif timestamp in cited:
        style += CITED_STYLE

    marker = ""
    if timestamp in cited:
        marker = (
            "<span style='color:#c98a2e;font-size:0.78em'> "
            f"[{', '.join(sorted(set(cited[timestamp])))}]</span>"
        )

    text = _escape(row["text"])
    speaker = _escape(row["speaker"])
    return (
        f"<div style='{style}'>"
        f"<span style='color:gray'>[{timestamp}]</span> "
        f"<b>{speaker}:</b> {text}{marker}</div>"
    )


def _escape(text: str) -> str:
    """Transcript text is model output; never let it inject markup."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
