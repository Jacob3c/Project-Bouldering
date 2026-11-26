"""Add the latest feedback flags to metadata/flag_reviews.csv for you to review.

Usage:
    python -m boulder.review.export

Run after boulder.render.feedback. New flags get a blank verdict; rows you already
filled in are kept, and unreviewed rows for flags that no longer exist are removed.
Then open the CSV in VS Code, watch each feedback video, and fill in verdict and note:
    correct  the detection is right (the arm was bent / the foot did readjust)
    wrong    it didn't happen (say why in note)
    unsure
and, for correct ones, mistake = yes or no (no = needed or harmless, e.g. a lock-off).
To add a mistake the system missed, add a row with verdict "missed", source "you",
the flag type (or "other") and the time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .flags import FEEDBACK_ROOT, REVIEW_PATH, load_events, merge_events, read_reviews, write_reviews


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reviews", type=Path, default=REVIEW_PATH)
    parser.add_argument("--feedback", type=Path, default=FEEDBACK_ROOT)
    args = parser.parse_args()

    events = load_events(args.feedback)
    if events.empty:
        parser.error("no feedback events found (run python -m boulder.render.feedback first)")
    merged, counts = merge_events(read_reviews(args.reviews), events)
    write_reviews(merged, args.reviews)
    todo = int((merged["verdict"] == "").sum())
    print(f"wrote {args.reviews}: {counts['added']} new flags, {counts['kept']} kept, "
          f"{counts['dropped']} stale removed; {todo} rows still need a verdict")


if __name__ == "__main__":
    main()
