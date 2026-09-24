"""Phase 1 deliverable: audio (or text) in -> canonical transcript out.

    python scripts/transcribe.py data/audio/standup.m4a
    python scripts/transcribe.py data/audio/standup.m4a --model small --out -
    python scripts/transcribe.py data/transcripts/team-sync-2026-09-08.txt --stats

Writes `data/transcripts/<meeting_id>.txt` unless --out says otherwise.
Passing an existing .txt just parses and re-renders it, which is how you
check that a Zoom or Teams export is in a shape the pipeline accepts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mat.cli import configure_stdout  # noqa: E402

from mat.config import TRANSCRIPT_DIR  # noqa: E402
from mat.ingest import (  # noqa: E402
    AudioIngestError,
    TranscriptFormatError,
    load_transcript_file,
    save_transcript,
    transcribe,
)


def _progress(fraction: float, text: str) -> None:
    bar = "=" * int(fraction * 30)
    preview = text[:48].replace("\n", " ")
    sys.stderr.write(f"\r  [{bar:<30}] {fraction:5.1%}  {preview:<50}")
    sys.stderr.flush()


def _report(transcript, label: str) -> None:
    print(f"\n{label}")
    print(f"  meeting_id : {transcript.meeting_id}")
    print(f"  source     : {transcript.source}")
    print(f"  segments   : {len(transcript)}")
    print(f"  words      : {transcript.word_count}")
    print(f"  duration   : {transcript.duration / 60:.1f} min")
    print(f"  speakers   : {', '.join(transcript.speakers)}")
    print(f"  est tokens : ~{transcript.word_count * 4 // 3}")


def main() -> int:
    configure_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="audio file, or an existing .txt transcript")
    parser.add_argument("--meeting-id", help="defaults to the input filename stem")
    parser.add_argument("--model", help="whisper model name (default from .env)")
    parser.add_argument("--out", help="output path, or '-' for stdout")
    parser.add_argument("--stats", action="store_true", help="print stats only, write nothing")
    args = parser.parse_args()

    meeting_id = args.meeting_id or args.input.stem

    try:
        if args.input.suffix.lower() == ".txt":
            transcript = load_transcript_file(args.input, meeting_id=meeting_id)
            _report(transcript, f"Parsed {args.input.name}")
        else:
            print(f"Transcribing {args.input.name} ...", file=sys.stderr)
            result = transcribe(
                args.input,
                meeting_id=meeting_id,
                model_name=args.model,
                on_progress=_progress,
            )
            sys.stderr.write("\n")
            transcript = result.transcript
            _report(transcript, f"Transcribed {args.input.name}")
            print(f"  language   : {result.language} ({result.language_probability:.2f})")
            for warning in result.warnings:
                print(f"  WARNING    : {warning}")
    except (AudioIngestError, TranscriptFormatError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.stats:
        return 0

    if args.out == "-":
        print()
        print(transcript.to_canonical_text())
        return 0

    destination = Path(args.out) if args.out else TRANSCRIPT_DIR / f"{meeting_id}.txt"
    save_transcript(transcript, destination)
    print(f"\n  written -> {destination}")

    if transcript.source == "audio":
        print(
            "\n  Speakers are UNKNOWN - Whisper does not identify them. "
            "Assign names in Phase 2, or edit the file by hand for now."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
