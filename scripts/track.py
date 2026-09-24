"""Phase 4 deliverable: multi-meeting history, queryable.

    python scripts/track.py ingest data/transcripts/client-status-2026-09-03.txt
    python scripts/track.py load data/extracts/team-sync-2026-09-08.json
    python scripts/track.py meetings
    python scripts/track.py tracker --as-of 2026-09-20
    python scripts/track.py people

`ingest` runs the full pipeline (extract then save). `load` stores an
extract that was produced earlier, which avoids paying for the same LLM
calls twice while working on the storage layer.
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
from mat.extract import MeetingExtract, extract_meeting  # noqa: E402
from mat.config import TRANSCRIPT_DIR  # noqa: E402
from mat.ingest import load_transcript_file  # noqa: E402
from mat.llm import get_client  # noqa: E402
from mat.store import (  # noqa: E402
    action_items,
    connect,
    delete_meeting,
    get_meeting,
    list_meetings,
    open_items_before,
    overdue_items,
    owner_workload,
    save_meeting,
)

DATE_IN_NAME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _meeting_date(argument: str | None, stem: str) -> date:
    if argument:
        return date.fromisoformat(argument)
    match = DATE_IN_NAME_RE.search(stem)
    if match:
        return date.fromisoformat(match.group(1))
    raise SystemExit("cannot determine the meeting date: pass --date YYYY-MM-DD")


def cmd_ingest(connection, args) -> int:
    transcript = load_transcript_file(args.path)
    meeting_date = _meeting_date(args.date, args.path.stem)

    # Re-ingesting is NOT reliably idempotent. Item ids derive from the
    # timestamp and wording the model chose, and those vary between runs
    # even at temperature 0 - so a second run inserts near-duplicates
    # rather than updating the first run's rows. Refuse by default.
    if get_meeting(connection, transcript.meeting_id) is not None:
        if not args.replace:
            print(
                f"'{transcript.meeting_id}' is already stored. Re-ingesting would add "
                "near-duplicate rows, because item ids depend on wording and timestamps "
                "the model picks fresh each run.\n"
                "  Use --replace to delete and re-extract it.",
                file=sys.stderr,
            )
            return 1
        delete_meeting(connection, transcript.meeting_id)
        print(f"  replaced existing '{transcript.meeting_id}'", file=sys.stderr)

    result = extract_meeting(
        transcript,
        client=get_client(),
        meeting_date=meeting_date,
        title=args.title,
        on_progress=lambda i, total: print(
            f"  chunk {i + 1}/{total} ...", file=sys.stderr, flush=True
        ),
    )
    print(f"  {result.report.summary_line()}", file=sys.stderr)

    report = save_meeting(connection, result.extract, transcript=transcript)
    print(report.summary_line())
    return 0


def cmd_load(connection, args) -> int:
    extract = MeetingExtract.model_validate(json.loads(args.path.read_text(encoding="utf-8")))

    # Attach the matching transcript when one is on disk. Without it the
    # stored items cite timestamps that cannot be looked up, which disables
    # the whole point of the citations.
    transcript = None
    source = args.transcript or TRANSCRIPT_DIR / f"{extract.meeting_id}.txt"
    if Path(source).exists():
        transcript = load_transcript_file(source, meeting_id=extract.meeting_id)
    else:
        print(f"  no transcript at {source}; citations will not be verifiable", file=sys.stderr)

    print(save_meeting(connection, extract, transcript=transcript).summary_line())
    return 0


def cmd_meetings(connection, _args) -> int:
    rows = list_meetings(connection)
    if not rows:
        print("no meetings stored yet")
        return 0

    header = f"{'date':<12}{'meeting':<34}{'actions':>9}{'open':>7}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['meeting_date']:<12}{(row['title'] or row['meeting_id'])[:33]:<34}"
            f"{row['action_count']:>9}{row['open_count']:>7}"
        )
    return 0


def cmd_tracker(connection, args) -> int:
    as_of = date.fromisoformat(args.as_of) if args.as_of else date.today()

    overdue = overdue_items(connection, as_of)
    if overdue:
        print(f"OVERDUE as of {as_of}")
        for row in overdue:
            print(f"  {row['due_date']}  {row['owner']:<18} {row['task']}")
            print(f"              from {row['meeting_id']} [{row['timestamp']}]")
        print()

    carried = open_items_before(connection, as_of)
    overdue_ids = {row["item_id"] for row in overdue}
    still_open = [row for row in carried if row["item_id"] not in overdue_ids]

    if still_open:
        print("OPEN")
        for row in still_open:
            due = row["due_date"] or "no date"
            carried_marker = "  (carried forward)" if row["carried_from"] else ""
            print(f"  {due:<12}{row['owner']:<18} {row['task']}{carried_marker}")

    if not overdue and not still_open:
        print("nothing outstanding")
    return 0


def cmd_people(connection, _args) -> int:
    rows = owner_workload(connection)
    if not rows:
        print("no action items stored yet")
        return 0

    header = f"{'owner':<24}{'total':>7}{'done':>7}{'open':>7}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['owner']:<24}{row['total']:>7}{row['done'] or 0:>7}{row['outstanding']:>7}")
    return 0


def cmd_items(connection, args) -> int:
    rows = action_items(connection, owner=args.owner, status=args.status)
    for row in rows:
        print(f"[{row['meeting_date']}] {row['owner']:<18} {row['status']:<12} {row['task']}")
    print(f"\n{len(rows)} item(s)")
    return 0


def main() -> int:
    configure_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", help="database path (default from .env)")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="extract a transcript and store it")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--date")
    ingest.add_argument("--title", default="")
    ingest.add_argument(
        "--replace", action="store_true", help="delete the stored meeting first"
    )
    ingest.set_defaults(handler=cmd_ingest)

    load = sub.add_parser("load", help="store an extract produced earlier")
    load.add_argument("path", type=Path)
    load.add_argument("--transcript", type=Path, help="transcript to attach")
    load.set_defaults(handler=cmd_load)

    sub.add_parser("meetings", help="list stored meetings").set_defaults(handler=cmd_meetings)

    tracker = sub.add_parser("tracker", help="outstanding work across all meetings")
    tracker.add_argument("--as-of", help="YYYY-MM-DD, defaults to today")
    tracker.set_defaults(handler=cmd_tracker)

    sub.add_parser("people", help="commitments per person").set_defaults(handler=cmd_people)

    items = sub.add_parser("items", help="filter action items")
    items.add_argument("--owner")
    items.add_argument("--status")
    items.set_defaults(handler=cmd_items)

    args = parser.parse_args()
    connection = connect(args.db)
    try:
        return args.handler(connection, args)
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
