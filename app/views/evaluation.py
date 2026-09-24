"""Accuracy against hand-labelled ground truth.

The tab exists so the number is visible, not buried in a script nobody
runs. Everything here is measured against labels a person wrote before
seeing any model output; the correction counts on the Feedback tab are a
different, weaker signal and are kept separate on purpose.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from mat.config import DATA_DIR, GROUND_TRUTH_DIR, TRANSCRIPT_DIR
from mat.eval import aggregate, load_ground_truth, score_meeting
from mat.extract import MeetingExtract
from mat.ingest import load_transcript_file

from ..components import empty_state

EXTRACT_DIR = DATA_DIR / "extracts"


def render(connection) -> None:
    st.subheader("Evaluation")

    labelled = sorted(
        path for path in GROUND_TRUTH_DIR.glob("*.json") if not path.name.startswith("_")
    )
    if not labelled:
        empty_state(
            "No hand-labelled meetings yet.",
            "Add labels to data/ground_truth/<meeting_id>.json using _template.json. "
            "The model cannot grade its own homework, so this stays empty until "
            "somebody writes labels by hand.",
        )
        return

    scores, skipped = [], []
    for path in labelled:
        extract_path = EXTRACT_DIR / f"{path.stem}.json"
        if not extract_path.exists():
            skipped.append(path.stem)
            continue
        extract = MeetingExtract.model_validate(
            json.loads(extract_path.read_text(encoding="utf-8"))
        )
        scores.append(
            score_meeting(extract, load_ground_truth(path), _valid_timestamps(path.stem))
        )

    if skipped:
        st.warning(
            "Labelled but not yet extracted: "
            + ", ".join(skipped)
            + ". Run `python scripts/extract.py <transcript> --json data/extracts/<id>.json`."
        )

    if not scores:
        return

    _headline(scores)
    _per_meeting(scores)
    _sections(scores)
    _errors(scores)


def _valid_timestamps(meeting_id: str):
    path = TRANSCRIPT_DIR / f"{meeting_id}.txt"
    if not path.exists():
        return None
    return frozenset(segment.timestamp for segment in load_transcript_file(path))


def _headline(scores) -> None:
    totals = aggregate(scores)["action_items"]
    owner_compared = sum(s.owner_compared for s in scores)
    owner_correct = sum(s.owner_correct for s in scores)
    date_compared = sum(s.date_compared for s in scores)
    date_correct = sum(s.date_correct for s in scores)
    unsupported = sum(s.unsupported for s in scores)

    columns = st.columns(5)
    columns[0].metric("Action item recall", f"{totals.recall:.0%}")
    columns[1].metric("Action item precision", f"{totals.precision:.0%}")
    columns[2].metric(
        "Owner accuracy",
        f"{owner_correct / owner_compared:.0%}" if owner_compared else "—",
        help="Over matched items only",
    )
    columns[3].metric(
        "Date accuracy",
        f"{date_correct / date_compared:.0%}" if date_compared else "—",
        help="Exact date match, over matched items only",
    )
    columns[4].metric(
        "Unsupported",
        unsupported,
        help="Items citing a transcript line that does not exist",
    )

    st.caption(
        f"Measured over {len(scores)} hand-labelled meeting(s). Recall is the "
        "share of real items found; precision is the share of extracted items "
        "that are real. A prediction matches a label when the text overlaps and "
        "the timestamps are close, one-to-one — so a duplicate costs a false "
        "positive rather than earning a second match."
    )


def _per_meeting(scores) -> None:
    st.markdown("### Per meeting")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Meeting": score.meeting_id,
                    "Predicted": score.actions.predicted,
                    "Labelled": score.actions.gold,
                    "Matched": score.actions.matched,
                    "Precision": f"{score.actions.precision:.0%}",
                    "Recall": f"{score.actions.recall:.0%}",
                    "F1": f"{score.actions.f1:.0%}",
                }
                for score in scores
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )


def _sections(scores) -> None:
    st.markdown("### By section")
    rows = [
        {
            "Section": name.replace("_", " "),
            "Predicted": section.predicted,
            "Labelled": section.gold,
            "Matched": section.matched,
            "Precision": f"{section.precision:.0%}",
            "Recall": f"{section.recall:.0%}",
        }
        for name, section in aggregate(scores).items()
        if section.gold or section.predicted
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _errors(scores) -> None:
    st.markdown("### What it got wrong")
    for score in scores:
        if not (score.missed_items or score.spurious_items):
            continue
        with st.expander(
            f"{score.meeting_id} — {len(score.missed_items)} missed, "
            f"{len(score.spurious_items)} spurious"
        ):
            if score.missed_items:
                st.markdown("**Missed** (in the labels, not extracted)")
                for text in score.missed_items:
                    st.markdown(f"- {text}")
            if score.spurious_items:
                st.markdown("**Spurious** (extracted, not in the labels)")
                for text in score.spurious_items:
                    st.markdown(f"- {text}")
