"""Compare MediaPipe and RTMPose on the same video(s).

Usage:
    python -m boulder.pose.compare                                  # mediapipe vs rtmpose
    python -m boulder.pose.compare --a mediapipe --b rtmpose-feet   # any two model folders

Only joints both models output (same name) are compared.

Prints, per shared joint:
    detected_a / detected_b      share of frames where each model found the joint
                                 with confidence >= 0.5
    median_gap_px                median pixel distance between the two models'
                                 positions for that joint (small = they agree)

Confidence scores are on different scales for the two models, so we compare
how often each is confident and where they put the joint, not the raw scores.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

QUERY = """
WITH a AS (
    SELECT * FROM read_parquet('{root}/model={a}/*.parquet')
), b AS (
    SELECT * FROM read_parquet('{root}/model={b}/*.parquet')
), both_models AS (
    SELECT
        a.keypoint,
        a.confidence AS conf_a,
        b.confidence AS conf_b,
        sqrt(power(a.x_px - b.x_px, 2) + power(a.y_px - b.y_px, 2)) AS gap_px
    FROM a
    JOIN b USING (video_id, frame_idx, keypoint)
)
SELECT
    keypoint,
    count(*)                                         AS frames,
    round(avg(CASE WHEN conf_a >= 0.5 THEN 1 ELSE 0 END), 2) AS detected_a,
    round(avg(CASE WHEN conf_b >= 0.5 THEN 1 ELSE 0 END), 2) AS detected_b,
    round(median(gap_px), 1)                         AS median_gap_px
FROM both_models
GROUP BY keypoint
ORDER BY median_gap_px DESC
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data/processed/keypoints"))
    parser.add_argument("--a", default="mediapipe", help="first model folder name")
    parser.add_argument("--b", default="rtmpose", help="second model folder name")
    args = parser.parse_args()
    print(f"a = {args.a}   b = {args.b}")
    print(duckdb.sql(QUERY.format(root=args.root.as_posix(), a=args.a, b=args.b)))


if __name__ == "__main__":
    main()
