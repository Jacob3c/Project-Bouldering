"""Compare your verdicts in metadata/flag_reviews.csv with the current feedback flags.

Usage:
    python -m boulder.review.score

Run after reviewing, and again after changing a threshold and re-rendering: a good
change lowers `wrong` without raising `correct_lost`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .flags import FEEDBACK_ROOT, REVIEW_PATH, clean_reviews, load_events, read_reviews, score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reviews", type=Path, default=REVIEW_PATH)
    parser.add_argument("--feedback", type=Path, default=FEEDBACK_ROOT)
    args = parser.parse_args()

    reviews = read_reviews(args.reviews)
    _, issues = clean_reviews(reviews)
    for issue in issues:
        print(f"issue: {issue}")
    table = score(reviews, load_events(args.feedback))
    if table.empty:
        print("nothing to score yet (run python -m boulder.review.export, then fill in verdicts)")
        return
    print(table.replace({np.nan: None}).to_string(index=False, float_format="{:.2f}".format))

    notes = reviews[(reviews["note"] != "") & reviews["verdict"].isin(["wrong", "missed"])]
    if len(notes):
        print("\nyour notes on wrong and missed flags:")
        for r in notes.itertuples(index=False):
            print(f"  {r.video_id}  {r.start}  {r.flag} ({r.verdict}): {r.note}")


if __name__ == "__main__":
    main()
