from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mat.store import connect  # noqa: E402

from app.views import (  # noqa: E402
    evaluation,
    feedback,
    meeting,
    tracker,
    transcript,
    upload,
)

TABS = ["Tracker", "Meeting", "Transcript", "Upload", "Feedback", "Evaluation"]


@st.cache_resource
def get_connection():
    """One connection for the app's lifetime.

    check_same_thread is disabled because Streamlit reruns may land on
    different script-runner threads; the app serialises its own writes.
    """
    return connect(check_same_thread=False)


def main() -> None:
    st.set_page_config(
        page_title="Meeting Action Tracker",
        page_icon="🗒️",
        layout="wide",
    )

    st.title("Meeting Action Tracker")
    st.caption(
        "Decisions, owner-tagged action items and unresolved questions, "
        "each traceable to the line of transcript it came from."
    )

    connection = get_connection()

    # A timestamp click on another tab asks to open the Transcript tab.
    # Streamlit cannot switch st.tabs programmatically, so the request is
    # surfaced as a prompt rather than pretended to work.
    if st.session_state.pop("active_tab", None) == "transcript":
        st.info("Line selected — open the **Transcript** tab to see it in context.")

    tabs = st.tabs(TABS)
    with tabs[0]:
        tracker.render(connection)
    with tabs[1]:
        meeting.render(connection)
    with tabs[2]:
        transcript.render(connection)
    with tabs[3]:
        upload.render(connection)
    with tabs[4]:
        feedback.render(connection)
    with tabs[5]:
        evaluation.render(connection)


if __name__ == "__main__":
    main()
