"""Tests for preparing keypoints before metrics: quality masking and the climb window."""

import pandas as pd
import pytest
from test_metrics import FPS, _keypoints, _skeleton

from boulder.transforms.metrics import attempt_metrics, frame_metrics
from boulder.transforms.prepare import (
    apply_climb_window,
    apply_point_quality,
    load_climb_windows,
)


def _kp(frames=(0, 2, 4), joints=("left_wrist", "right_wrist")):
    return pd.DataFrame([(f, f * 33, j, 0.9) for f in frames for j in joints],
                        columns=["frame_idx", "timestamp_ms", "keypoint", "confidence"])


def test_bad_points_get_zero_confidence_and_rows_are_kept():
    kp = _kp()
    quality = kp[["frame_idx", "keypoint"]].assign(point_ok=True, valid=True)
    quality.loc[(quality.frame_idx == 2) & (quality.keypoint == "left_wrist"), "point_ok"] = False
    quality.loc[quality.frame_idx == 4, ["point_ok", "valid"]] = False     # whole frame invalid

    out, invalid_frames = apply_point_quality(kp, quality)
    assert len(out) == len(kp)                                             # nothing deleted
    zeroed = out[out.confidence == 0][["frame_idx", "keypoint"]].values.tolist()
    assert zeroed == [[2, "left_wrist"], [4, "left_wrist"], [4, "right_wrist"]]
    assert invalid_frames == 1
    assert list(out.columns) == list(kp.columns)


def test_points_missing_from_quality_file_are_left_alone():
    kp = _kp(frames=(0, 2))
    quality = kp[kp.frame_idx == 0][["frame_idx", "keypoint"]].assign(point_ok=False, valid=True)
    out, _ = apply_point_quality(kp, quality)
    assert (out[out.frame_idx == 2].confidence == 0.9).all()


def test_climb_window_keeps_only_frames_inside():
    kp = _kp(frames=range(0, 200, 2))                      # 0 .. 6.5 s
    out = apply_climb_window(kp, 1.0, 3.0)
    t = out.timestamp_ms / 1000
    assert t.min() >= 1.0 and t.max() <= 3.0
    assert apply_climb_window(kp, None, None).equals(kp)   # blank window = whole clip
    assert (apply_climb_window(kp, None, 2.0).timestamp_ms <= 2000).all()


def test_duration_counts_from_window_start():
    seq = [_skeleton() for _ in range(4 * FPS)]             # 4 s clip
    kp = apply_climb_window(_keypoints(seq), 1.0, 3.0)
    row = attempt_metrics(frame_metrics(kp))
    assert row["duration_s"] == pytest.approx(2.0, abs=0.1)  # not 3.0 (the end timestamp)


def test_load_climb_windows(tmp_path):
    pytest.importorskip("pyarrow")
    path = tmp_path / "clips.parquet"
    pd.DataFrame({"clip_id": ["a", "b"], "climb_start_s": [4.2, None],
                  "climb_end_s": [36.8, None]}).to_parquet(path)
    assert load_climb_windows(path) == {"a": (4.2, 36.8), "b": (None, None)}
    assert load_climb_windows(tmp_path / "missing.parquet") == {}
