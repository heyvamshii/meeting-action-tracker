"""Upload a recording or transcript and run the pipeline.

Both entry points end in the same place. The text path exists because
transcription on CPU is slow and because a Zoom or Teams export is already
better than anything Whisper will produce - it carries real speaker labels.
"""

from __future__ import annotations

from datetime import date

import streamlit as st

from mat.config import AUDIO_DIR, TRANSCRIPT_DIR, settings
from mat.extract import collect_feedback, extract_meeting
from mat.ingest import (
    AudioIngestError,
    SUPPORTED_SUFFIXES,
    TranscriptFormatError,
    attribute_speakers,
    parse_transcript_text,
    save_transcript,
    transcribe,
)
from mat.llm import LLMUnavailableError, get_client, try_get_client
from mat.store import save_meeting

TEXT_SUFFIXES = ("txt", "md", "vtt")


def render(connection) -> None:
    st.subheader("Add a meeting")
    _show_last_result()

    if try_get_client() is None:
        st.error(
            f"No LLM provider is reachable (LLM_PROVIDER={settings.llm_provider}). "
            "Set GROQ_API_KEY in .env, or run `ollama serve` and set LLM_PROVIDER=ollama."
        )
        return

    mode = st.radio(
        "Source",
        ["Transcript (text)", "Recording (audio)"],
        horizontal=True,
        help="Text is faster and keeps real speaker labels. Audio needs transcribing first.",
    )

    left, right = st.columns(2)
    meeting_date = left.date_input("Meeting date", value=date.today())
    title = right.text_input("Title", placeholder="Weekly client status")

    participants = st.text_input(
        "Participants (comma separated)",
        placeholder="Priya (PM), Arun (Dev), Mark (Client)",
        help="Used to attribute speakers. Attribution can decline, but it cannot "
        "invent a name outside this list.",
    )

    if mode.startswith("Transcript"):
        _text_flow(connection, meeting_date, title, participants)
    else:
        _audio_flow(connection, meeting_date, title, participants)


def _text_flow(connection, meeting_date, title, participants) -> None:
    uploaded = st.file_uploader("Transcript", type=list(TEXT_SUFFIXES))
    pasted = st.text_area(
        "…or paste it",
        height=180,
        placeholder="[00:00:04] Priya (PM): Where are we on the payment gateway?",
        help="One utterance per line, as [HH:MM:SS] Speaker: text",
    )

    text = uploaded.read().decode("utf-8", errors="replace") if uploaded else pasted
    default_id = uploaded.name.rsplit(".", 1)[0] if uploaded else f"meeting-{meeting_date}"
    meeting_id = st.text_input("Meeting id", value=default_id)

    if not st.button("Extract", type="primary", disabled=not text.strip()):
        return

    try:
        transcript = parse_transcript_text(text, meeting_id=meeting_id)
    except TranscriptFormatError as exc:
        st.error(f"Could not parse the transcript — {exc}")
        st.caption("Every line must look like: `[HH:MM:SS] Speaker: text`")
        return

    save_transcript(transcript, TRANSCRIPT_DIR / f"{meeting_id}.txt")
    _attribute_and_extract(connection, transcript, meeting_date, title, participants)


def _audio_flow(connection, meeting_date, title, participants) -> None:
    uploaded = st.file_uploader(
        "Recording", type=[suffix.lstrip(".") for suffix in sorted(SUPPORTED_SUFFIXES)]
    )
    if uploaded is None:
        return

    meeting_id = st.text_input("Meeting id", value=uploaded.name.rsplit(".", 1)[0])

    backend = st.radio(
        "Transcription",
        ["Hosted (accurate)", "Local (offline)"],
        horizontal=True,
        index=0 if settings.transcription_backend == "groq" else 1,
        help="Hosted uses Groq whisper-large-v3 — far better on real meeting "
        "audio, accents and background noise. Local runs on your CPU with no "
        "network but degrades badly on difficult recordings.",
    )
    use_groq = backend.startswith("Hosted")

    if use_groq:
        st.caption(
            f"Uses Groq '{settings.groq_whisper_model}'. Files over 24 MB have their "
            "audio track extracted first, so a large screen recording uploads as a "
            "few MB. Falls back to local if the service is unreachable."
        )
    else:
        st.caption(
            f"Runs on CPU with the local '{settings.whisper_model}' model — roughly "
            "1–2 minutes per 10 minutes of audio, and noticeably weaker on accented "
            "or noisy speech."
        )
    st.caption("Neither identifies speakers — those are resolved separately.")

    if not st.button("Transcribe and extract", type="primary"):
        return

    destination = AUDIO_DIR / uploaded.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(uploaded.getbuffer())

    progress = st.progress(0.0, text="Loading the model…")

    try:
        result = transcribe(
            destination,
            meeting_id=meeting_id,
            backend="groq" if use_groq else "local",
            on_progress=lambda fraction, text: progress.progress(
                fraction, text=f"Transcribing… {text[:60]}"
            ),
        )
    except AudioIngestError as exc:
        progress.empty()
        st.error(f"Could not transcribe — {exc}")
        return

    progress.empty()
    if use_groq and result.backend == "local":
        st.warning("Hosted transcription was unavailable; fell back to the local model.")
    st.caption(
        f"{len(result.transcript)} lines · {result.transcript.word_count} words · "
        f"{result.words_per_minute:.0f} words/min · {result.backend}"
    )
    for warning in result.warnings:
        st.warning(warning)

    save_transcript(result.transcript, TRANSCRIPT_DIR / f"{meeting_id}.txt")
    _attribute_and_extract(
        connection, result.transcript, meeting_date, title, participants
    )


def _attribute_and_extract(connection, transcript, meeting_date, title, participants) -> None:
    names = [name.strip() for name in participants.split(",") if name.strip()]

    with st.status("Working…", expanded=True) as status:
        st.write("Attributing speakers")
        attribution = attribute_speakers(transcript, names or None, client=try_get_client())
        transcript = attribution.transcript
        st.write(
            f"  {attribution.method}: {attribution.assigned}/{len(transcript)} lines named "
            f"({attribution.coverage:.0%})"
        )

        examples = collect_feedback(connection, meeting_id=transcript.meeting_id)
        if examples:
            st.write(
                f"  applying {len(examples.rejected)} rejected and "
                f"{len(examples.added)} added examples from earlier corrections"
            )

        st.write("Extracting items")
        try:
            result = extract_meeting(
                transcript,
                client=get_client(),
                meeting_date=meeting_date,
                title=title,
                on_progress=lambda index, total: st.write(f"  chunk {index + 1}/{total}"),
                feedback=examples or None,
            )
        except LLMUnavailableError as exc:
            status.update(label="Extraction failed", state="error")
            st.error(str(exc))
            return

        st.write(f"  {result.report.summary_line()}")
        for note in result.report.notes[:5]:
            st.write(f"  note: {note}")

        st.write("Saving")
        report = save_meeting(connection, result.extract, transcript=transcript)
        st.write(f"  {report.summary_line()}")

        status.update(label="Done", state="complete")

    # Streamlit renders every tab top to bottom in one pass, and the Meeting
    # tab is rendered BEFORE this one. Without an explicit rerun the Meeting
    # tab keeps showing whichever meeting was selected when the pass began -
    # which looks exactly like the upload having been ignored.
    st.session_state["selected_meeting"] = result.extract.meeting_id
    st.session_state["just_extracted"] = {
        "meeting_id": result.extract.meeting_id,
        "items": result.extract.item_count,
        "chunk_failures": result.report.chunk_failures,
        "chunks": result.report.chunks,
    }
    st.rerun()


def _show_last_result() -> None:
    """Report the previous upload, after the rerun that switched meetings."""
    outcome = st.session_state.pop("just_extracted", None)
    if not outcome:
        return

    if outcome["chunk_failures"]:
        st.warning(
            f"{outcome['chunk_failures']} of {outcome['chunks']} chunks failed. "
            "Part of this meeting is missing."
        )

    if outcome["items"] == 0:
        st.error(
            f"**{outcome['meeting_id']}: nothing was extracted.** The transcript "
            "was processed but contained no decisions, action items, questions "
            "or risks that could be tied to a transcript line.\n\n"
            "Usually this means the transcript itself is poor — check the "
            "Transcript tab. If it reads as fragments rather than sentences, "
            "transcription failed, not extraction."
        )
        return

    st.success(
        f"**{outcome['meeting_id']}** is now selected on the Meeting tab — "
        f"{outcome['items']} items stored."
    )
