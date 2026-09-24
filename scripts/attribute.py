"""Phase 2 deliverable: assign speakers to transcript segments.

    python scripts/attribute.py data/transcripts/standup-eng-2026-09-04.txt \
        --participants "Ravi,Deepa,Karthik,Aditi,Suresh"

    python scripts/attribute.py <file> --participants "..." --llm
    python scripts/attribute.py --benchmark

--benchmark strips the speakers off every committed transcript, re-derives
them, and scores the result against the originals. That number is the
honest measure of what attribution is worth, and it is what belongs in the
write-up rather than a claim that it "works".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mat.cli import configure_stdout  # noqa: E402

from mat.config import TRANSCRIPT_DIR  # noqa: E402
from mat.ingest import (  # noqa: E402
    Roster,
    UNKNOWN_SPEAKER,
    attribute_speakers,
    load_transcript_file,
    save_transcript,
)
from mat.llm import try_get_client  # noqa: E402


def _strip(transcript):
    return transcript.with_segments(
        segment.with_speaker(UNKNOWN_SPEAKER) for segment in transcript
    )


def _score(predicted, truth) -> tuple[int, int, int]:
    """(correct, wrong, abstained) over segments."""
    correct = wrong = abstained = 0
    for guess, actual in zip(predicted, truth):
        if guess.speaker == UNKNOWN_SPEAKER:
            abstained += 1
        elif guess.speaker == actual.speaker:
            correct += 1
        else:
            wrong += 1
    return correct, wrong, abstained


def benchmark(use_llm: bool) -> int:
    client = try_get_client() if use_llm else None
    if use_llm and client is None:
        print("no LLM available; benchmarking the heuristic path only\n")

    header = f"{'meeting':<32}{'segs':>6}{'correct':>9}{'wrong':>7}{'abstain':>9}{'precision':>11}"
    print(header)
    print("-" * len(header))

    totals = [0, 0, 0, 0]

    for path in sorted(TRANSCRIPT_DIR.glob("*.txt")):
        truth = load_transcript_file(path)
        roster = Roster.from_transcript(truth)

        result = attribute_speakers(_strip(truth), roster.labels, client=client)
        correct, wrong, abstained = _score(result.transcript, truth)

        decided = correct + wrong
        precision = correct / decided if decided else 0.0
        print(
            f"{path.stem:<32}{len(truth):>6}{correct:>9}{wrong:>7}"
            f"{abstained:>9}{precision:>10.0%}"
        )

        totals = [
            totals[0] + len(truth),
            totals[1] + correct,
            totals[2] + wrong,
            totals[3] + abstained,
        ]

    segments, correct, wrong, abstained = totals
    decided = correct + wrong
    print("-" * len(header))
    print(
        f"{'TOTAL':<32}{segments:>6}{correct:>9}{wrong:>7}{abstained:>9}"
        f"{(correct / decided if decided else 0):>10.0%}"
    )
    print(f"\ncoverage (lines given a name): {decided / segments:.0%}" if segments else "")
    print("precision is over lines the system was willing to name; abstentions are not errors.")
    return 0


def main() -> int:
    configure_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", help="a .txt transcript")
    parser.add_argument("--participants", help="comma-separated names, e.g. 'Ravi,Deepa'")
    parser.add_argument("--llm", action="store_true", help="use the configured LLM provider")
    parser.add_argument("--benchmark", action="store_true", help="score against the corpus")
    parser.add_argument("--out", help="write the attributed transcript here")
    args = parser.parse_args()

    if args.benchmark:
        return benchmark(use_llm=args.llm)

    if not args.input:
        parser.error("give a transcript, or use --benchmark")

    transcript = load_transcript_file(args.input)
    participants = (
        [p.strip() for p in args.participants.split(",") if p.strip()]
        if args.participants
        else None
    )

    client = None
    if args.llm:
        client = try_get_client()
        if client is None:
            print("warning: no LLM available, falling back to heuristics", file=sys.stderr)

    result = attribute_speakers(transcript, participants, client=client)

    print(f"\n{args.input.name}")
    print(f"  method   : {result.method}")
    print(f"  assigned : {result.assigned}/{len(result.transcript)} ({result.coverage:.0%})")
    print(f"  unknown  : {result.unknown}")
    print(f"  speakers : {', '.join(result.transcript.speakers)}")
    for note in result.notes:
        print(f"  note     : {note}")

    if args.out:
        save_transcript(result.transcript, args.out)
        print(f"\n  written -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
