"""Phase 3 deliverable: transcript in -> validated structured extract out.

    python scripts/extract.py data/transcripts/client-status-2026-09-03.txt
    python scripts/extract.py <file> --date 2026-09-11 --json out.json
    python scripts/extract.py <file> --no-summary --chunk-tokens 800

The meeting date is taken from a trailing YYYY-MM-DD in the filename when
not given, since relative deadlines cannot be resolved without it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mat.cli import configure_stdout  # noqa: E402

from mat.extract import extract_meeting  # noqa: E402
from mat.ingest import load_transcript_file  # noqa: E402
from mat.llm import get_client  # noqa: E402

DATE_IN_NAME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _meeting_date(argument: str | None, stem: str) -> date:
    if argument:
        return date.fromisoformat(argument)
    match = DATE_IN_NAME_RE.search(stem)
    if match:
        return date.fromisoformat(match.group(1))
    raise SystemExit(
        "cannot determine the meeting date: pass --date YYYY-MM-DD "
        "(relative deadlines like 'by Friday' need it)"
    )


def _print_extract(extract) -> None:
    print(f"\n{'=' * 72}\n{extract.title or extract.meeting_id}  ({extract.meeting_date})\n{'=' * 72}")

    if extract.summary:
        print(f"\n{extract.summary}\n")

    if extract.decisions:
        print("DECISIONS")
        for item in extract.decisions:
            agreed = f" [agreed: {', '.join(item.agreed_by)}]" if item.agreed_by else ""
            print(f"  [{item.timestamp}] {item.decision}")
            print(f"             by {item.made_by}{agreed}  (conf {item.confidence:.2f})")

    if extract.action_items:
        print("\nACTION ITEMS")
        for item in extract.action_items:
            due = item.due_date.isoformat() if item.due_date else "no date"
            flag = " <-- verify" if item.confidence < 0.8 else ""
            print(f"  [{item.timestamp}] {item.task}")
            print(
                f"             owner {item.owner} ({item.owner_source.value})"
                f"  due {due} ({item.due_date_source.value})"
                f"  conf {item.confidence:.2f}{flag}"
            )
            if item.due_phrase:
                print(f'             heard: "{item.due_phrase}"')

    if extract.open_questions:
        print("\nOPEN QUESTIONS")
        for item in extract.open_questions:
            print(f"  [{item.timestamp}] {item.question}")
            print(f"             raised by {item.raised_by}, {item.status.value}")

    if extract.risks:
        print("\nRISKS")
        for item in extract.risks:
            print(f"  [{item.timestamp}] ({item.severity.value}) {item.risk}")


def main() -> int:
    configure_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="a .txt transcript")
    parser.add_argument("--date", help="meeting date, YYYY-MM-DD")
    parser.add_argument("--title", default="")
    parser.add_argument("--json", type=Path, help="also write the extract as JSON")
    parser.add_argument("--chunk-tokens", type=int, help="override the window size")
    parser.add_argument("--no-summary", action="store_true")
    args = parser.parse_args()

    transcript = load_transcript_file(args.input)
    meeting_date = _meeting_date(args.date, args.input.stem)

    print(
        f"{args.input.name}: {len(transcript)} segments, "
        f"~{transcript.word_count * 4 // 3} tokens",
        file=sys.stderr,
    )

    result = extract_meeting(
        transcript,
        client=get_client(),
        meeting_date=meeting_date,
        title=args.title,
        max_tokens=args.chunk_tokens,
        write_summary=not args.no_summary,
        on_progress=lambda i, total: print(
            f"  chunk {i + 1}/{total} ...", file=sys.stderr, flush=True
        ),
    )

    _print_extract(result.extract)

    print(f"\n{'-' * 72}")
    print(result.report.summary_line())
    for note in result.report.notes[:10]:
        print(f"  note: {note}")
    if len(result.report.notes) > 10:
        print(f"  ... and {len(result.report.notes) - 10} more")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(result.extract.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
        print(f"\n  written -> {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
