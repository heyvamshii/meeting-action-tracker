"""Phase 7 deliverable: score extractions against the hand labels.

    python scripts/evaluate.py                 # every labelled meeting
    python scripts/evaluate.py --meeting client-status-2026-09-03
    python scripts/evaluate.py --json data/eval/report.json

Reads extracts from data/extracts/ and labels from data/ground_truth/. A
meeting is only scored when both exist, and the ones that were skipped are
named rather than quietly ignored.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mat.cli import configure_stdout  # noqa: E402
from mat.config import DATA_DIR, GROUND_TRUTH_DIR, TRANSCRIPT_DIR  # noqa: E402
from mat.eval import aggregate, load_ground_truth, score_meeting  # noqa: E402
from mat.extract import MeetingExtract  # noqa: E402
from mat.ingest import load_transcript_file  # noqa: E402

EXTRACT_DIR = DATA_DIR / "extracts"


def _valid_timestamps(meeting_id: str) -> frozenset[str] | None:
    path = TRANSCRIPT_DIR / f"{meeting_id}.txt"
    if not path.exists():
        return None
    return frozenset(s.timestamp for s in load_transcript_file(path))


def main() -> int:
    configure_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meeting", help="score one meeting only")
    parser.add_argument("--json", type=Path, help="also write the report as JSON")
    parser.add_argument("--verbose", action="store_true", help="list misses and false positives")
    args = parser.parse_args()

    labelled = sorted(p for p in GROUND_TRUTH_DIR.glob("*.json") if not p.name.startswith("_"))
    if args.meeting:
        labelled = [p for p in labelled if p.stem == args.meeting]

    if not labelled:
        print("No hand-labelled meetings found in data/ground_truth/.")
        print("Label a meeting first - the model cannot grade its own homework.")
        return 1

    scores, skipped = [], []

    for path in labelled:
        extract_path = EXTRACT_DIR / f"{path.stem}.json"
        if not extract_path.exists():
            skipped.append(f"{path.stem}: no extract at {extract_path}")
            continue

        extract = MeetingExtract.model_validate(
            json.loads(extract_path.read_text(encoding="utf-8"))
        )
        scores.append(
            score_meeting(extract, load_ground_truth(path), _valid_timestamps(path.stem))
        )

    if skipped:
        print("Skipped (no extract to score):")
        for line in skipped:
            print(f"  {line}")
        print("  run: python scripts/extract.py <transcript> --json data/extracts/<id>.json\n")

    if not scores:
        return 1

    _print_report(scores, args.verbose)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(_as_dict(scores), indent=2), encoding="utf-8")
        print(f"\n  written -> {args.json}")

    return 0


def _print_report(scores, verbose: bool) -> None:
    header = f"{'meeting':<30}{'pred':>6}{'gold':>6}{'match':>7}{'prec':>7}{'rec':>7}{'F1':>7}"
    print("ACTION ITEMS")
    print(header)
    print("-" * len(header))

    for score in scores:
        actions = score.actions
        print(
            f"{score.meeting_id[:29]:<30}{actions.predicted:>6}{actions.gold:>6}"
            f"{actions.matched:>7}{actions.precision:>6.0%}{actions.recall:>6.0%}"
            f"{actions.f1:>6.0%}"
        )

    totals = aggregate(scores)
    print("-" * len(header))
    for section, total in totals.items():
        if not total.gold and not total.predicted:
            continue
        print(
            f"{section.replace('_', ' '):<30}{total.predicted:>6}{total.gold:>6}"
            f"{total.matched:>7}{total.precision:>6.0%}{total.recall:>6.0%}{total.f1:>6.0%}"
        )

    owner_compared = sum(s.owner_compared for s in scores)
    owner_correct = sum(s.owner_correct for s in scores)
    date_compared = sum(s.date_compared for s in scores)
    date_correct = sum(s.date_correct for s in scores)
    unsupported = sum(s.unsupported for s in scores)

    print("\nON MATCHED ACTION ITEMS")
    if owner_compared:
        print(f"  owner accuracy : {owner_correct / owner_compared:.0%} "
              f"({owner_correct}/{owner_compared})")
    if date_compared:
        print(f"  date accuracy  : {date_correct / date_compared:.0%} "
              f"({date_correct}/{date_compared})")
    print(f"  unsupported    : {unsupported} item(s) citing a line not in the transcript")

    if verbose:
        for score in scores:
            if not (score.missed_items or score.spurious_items):
                continue
            print(f"\n{score.meeting_id}")
            for text in score.missed_items:
                print(f"  MISSED    {text[:90]}")
            for text in score.spurious_items:
                print(f"  SPURIOUS  {text[:90]}")


def _as_dict(scores) -> dict:
    return {
        "meetings": [
            {
                "meeting_id": score.meeting_id,
                "sections": {
                    name: {
                        "predicted": section.predicted,
                        "gold": section.gold,
                        "matched": section.matched,
                        "precision": round(section.precision, 3),
                        "recall": round(section.recall, 3),
                        "f1": round(section.f1, 3),
                    }
                    for name, section in score.sections.items()
                },
                "owner_accuracy": round(score.owner_accuracy, 3),
                "date_accuracy": round(score.date_accuracy, 3),
                "unsupported": score.unsupported,
                "missed": score.missed_items,
                "spurious": score.spurious_items,
            }
            for score in scores
        ],
        "totals": {
            name: {
                "predicted": section.predicted,
                "gold": section.gold,
                "matched": section.matched,
                "precision": round(section.precision, 3),
                "recall": round(section.recall, 3),
                "f1": round(section.f1, 3),
            }
            for name, section in aggregate(scores).items()
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
