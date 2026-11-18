"""Prepare keypoints for metrics: drop what the quality checks rejected, keep only the climb.

Two steps, each optional:

quality   points the quality checks marked bad (point_ok = False, which includes
          every point in an invalid frame) get confidence 0. Metrics already treat
          low-confidence joints as missing, and keeping the rows keeps time gaps
          visible, so a long bad stretch is never bridged by interpolation.
window    only frames between climb_start_s and climb_end_s from the attempts log
          (data/clips.parquet) are kept. A blank start or end means "from the
          beginning" / "to the end" of the clip.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

CLIPS_PATH = Path("data/clips.parquet")


def apply_point_quality(keypoints: pd.DataFrame, quality_points: pd.DataFrame
                        ) -> tuple[pd.DataFrame, int]:
    """Set confidence to 0 where the quality checks said point_ok is False.

    Returns the keypoints and how many frames were fully invalid.
    """
    flags = quality_points[["frame_idx", "keypoint", "point_ok", "valid"]]
    merged = keypoints.merge(flags, on=["frame_idx", "keypoint"], how="left",
                             validate="one_to_one")
    bad = merged["point_ok"].eq(False)          # rows missing from the quality file stay as-is
    merged.loc[bad, "confidence"] = 0.0
    invalid_frames = int(merged.loc[merged["valid"].eq(False), "frame_idx"].nunique())
    return merged.drop(columns=["point_ok", "valid"]), invalid_frames


def apply_climb_window(keypoints: pd.DataFrame, start_s: float | None,
                       end_s: float | None) -> pd.DataFrame:
    """Keep frames whose timestamp lies inside [start_s, end_s]; None = open end."""
    t = keypoints["timestamp_ms"] / 1000
    keep = pd.Series(True, index=keypoints.index)
    if start_s is not None:
        keep &= t >= start_s
    if end_s is not None:
        keep &= t <= end_s
    return keypoints[keep]


def load_climb_windows(path: Path = CLIPS_PATH) -> dict[str, tuple[float | None, float | None]]:
    """{clip_id: (start_s, end_s)} from data/clips.parquet; blanks become None."""
    if not path.exists():
        return {}
    clips = pd.read_parquet(path)
    if "climb_start_s" not in clips.columns:
        return {}

    def clean(v) -> float | None:
        return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)

    return {row.clip_id: (clean(row.climb_start_s), clean(row.climb_end_s))
            for row in clips.itertuples(index=False)}
