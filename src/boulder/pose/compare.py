"""Compare MediaPipe and RTMPose on the same video(s).

Usage:
    python -m boulder.pose.compare            # every video processed by both models

Prints, per shared joint:
    detected_mp / detected_rtm   share of frames where each model found the joint
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
WITH mp AS (
    SELECT * FROM read_parquet('{root}/model=mediapipe/*.parquet')
), rtm AS (
    SELECT * FROM read_parquet('{root}/model=rtmpose/*.parquet')
), both_models AS (
    SELECT
        mp.keypoint,
        mp.confidence AS conf_mp,
        rtm.confidence AS conf_rtm,
        sqrt(power(mp.x_px - rtm.x_px, 2) + power(mp.y_px - rtm.y_px, 2)) AS gap_px
    FROM mp
    JOIN rtm USING (video_id, frame_idx, keypoint)
)
SELECT
    keypoint,
    count(*)                                         AS frames,
    round(avg(CASE WHEN conf_mp  >= 0.5 THEN 1 ELSE 0 END), 2) AS detected_mp,
    round(avg(CASE WHEN conf_rtm >= 0.5 THEN 1 ELSE 0 END), 2) AS detected_rtm,
    round(median(gap_px), 1)                         AS median_gap_px
FROM both_models
GROUP BY keypoint
ORDER BY median_gap_px DESC
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data/processed/keypoints"))
    args = parser.parse_args()
    print(duckdb.sql(QUERY.format(root=args.root.as_posix())))


if __name__ == "__main__":
    main()
