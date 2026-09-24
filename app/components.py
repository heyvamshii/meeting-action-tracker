from __future__ import annotations

import streamlit as st

from mat.config import settings

VERIFY = "⚠️"

OWNER_SOURCE_HELP = {
    "explicit": "named aloud in the transcript",
    "speaker": "inferred from who was speaking — attribution is ~63% accurate, verify",
    "none": "nobody took this item",
}

DUE_SOURCE_HELP = {
    "explicit": "a specific day was named",
    "inferred": "a vague period was named and its end was used — verify",
    "none": "no deadline was given",
}


def needs_verification(row) -> bool:
    """True when a human should look before acting on this item."""
    return (
        row["confidence"] < settings.low_confidence_threshold
        or row["owner_source"] == "speaker"
        or row["due_date_source"] == "inferred"
    )


def confidence_caption(row) -> str:
    parts = [f"conf {row['confidence']:.2f}"]
    if "owner_source" in row.keys():
        parts.append(f"owner: {row['owner_source']}")
    if "due_date_source" in row.keys() and row["due_date_source"] != "none":
        parts.append(f"due: {row['due_date_source']}")
    return " · ".join(parts)


def timestamp_button(row, key: str) -> None:
    """Jump the transcript tab to the line this item came from."""
    if st.button(f"`{row['timestamp']}`", key=key, help="show this line in the transcript"):
        st.session_state["jump_to"] = row["timestamp"]
        st.session_state["jump_meeting"] = row["meeting_id"]
        st.session_state["active_tab"] = "transcript"
        st.rerun()


def verification_legend() -> None:
    st.caption(
        f"{VERIFY} means a value was inferred rather than stated — an owner taken from "
        "the speaker label, a deadline read from a vague phrase, or low model "
        "confidence. Check these before acting on them."
    )


def empty_state(message: str, hint: str = "") -> None:
    st.info(message)
    if hint:
        st.caption(hint)
