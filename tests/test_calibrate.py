"""Calibration: features and grid search on a synthetic clip with known labels."""

import numpy as np
import pandas as pd

from boulder.review.calibrate import auc, evaluate, feature_summary, feature_table, grid_search
from boulder.transforms.metrics import MetricConfig, frame_metrics
from test_feedback import FPS, _keypoints, _skeleton

V = "2026-11-23_MainRec_Orange-V5-6_att01_send_handheld"


def _clip():
    """A real shuffle at ~1.0 s and an out-and-back jitter at ~2.3 s on the left foot."""
    seq = [_skeleton() for _ in range(FPS)]
    seq += [_skeleton(left_foot=(450, 1300 - 10 * k)) for k in range(1, 5)]       # 40 px = 0.2 TL
    seq += [_skeleton(left_foot=(450, 1260)) for _ in range(FPS)]
    # jitter: the foot point jumps 60 px out and straight back (pose noise, no real move)
    seq += [_skeleton(left_foot=(450 + d, 1260)) for d in (20, 40, 60, 60, 40, 20)]
    seq += [_skeleton(left_foot=(450, 1260)) for _ in range(FPS)]
    return frame_metrics(_keypoints(seq))


def _labels(frames):
    cfg = MetricConfig()
    from boulder.render.feedback import feedback_flags, flag_events
    ev = flag_events(feedback_flags(frames, cfg))
    ev = ev[ev["flag"] == "left_foot_readjust"].reset_index(drop=True)
    assert len(ev) == 2  # current settings flag both
    return pd.DataFrame({"video_id": V, "flag": ev["flag"], "start_s": ev["start_s"],
                         "end_s": ev["end_s"], "verdict": ["correct", "wrong"]})


def test_auc():
    assert auc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert auc(np.array([1.0]), np.array([1.0])) == 0.5


def test_features_separate_shuffle_from_jitter():
    frames = _clip()
    labels = _labels(frames)
    table = feature_table(labels, {V: frames})
    assert len(table) == 2
    summary = feature_summary(table).set_index("feature")
    assert summary.loc["gap_moved_tl", "auc"] == 1.0  # the real shuffle moved further


def test_grid_finds_setting_that_drops_jitter():
    frames = _clip()
    labels = _labels(frames)
    now = evaluate(MetricConfig(), labels, {V: frames})
    assert (now["correct_kept"], now["wrong_shown"]) == (1, 1)
    grid = {"foot_move_speed": [1.0], "foot_still_s": [0.5], "readjust_max_dist": [0.3],
            "foot_dominance": [1.0], "move_min_dist": [0.0, 0.1]}
    table = grid_search(MetricConfig(), labels, {V: frames}, grid)
    best = table.iloc[0]
    assert (best["correct_kept"], best["wrong_shown"]) == (1, 0)
    assert (table["current"] == "<- now").sum() == 1


def test_splits_are_stable_and_overridable(tmp_path):
    from boulder.review.splits import default_split, load_overrides, split_of

    assert default_split(V) == default_split(V)
    ids = [f"clip{i}" for i in range(400)]
    share = sum(default_split(i) == "test" for i in ids) / len(ids)
    assert 0.15 < share < 0.35
    p = tmp_path / "splits.csv"
    p.write_text(f"clip_id,split\n{V},test\nother,bogus\n")
    over = load_overrides(p)
    assert over == {V: "test"} and split_of(V, over) == "test"


def test_arm_grid_runs_and_marks_current():
    from boulder.review.calibrate import arm_grid_search
    from boulder.render.feedback import feedback_flags, flag_events

    seq = [_skeleton(bent=True) for _ in range(3 * FPS)]
    frames = frame_metrics(_keypoints(seq))
    ev = flag_events(feedback_flags(frames))
    ev = ev[ev["flag"] == "left_arm_bent"]
    labels = pd.DataFrame({"video_id": V, "flag": ev["flag"], "start_s": ev["start_s"],
                           "end_s": ev["end_s"], "verdict": "correct", "mistake": "no"})
    table = arm_grid_search(labels, {V: frames})
    assert (table["current"] == "<- now").sum() == 1
    now = table[table["current"] != ""].iloc[0]
    assert (now["correct_kept"], now["not_mistake_shown"], now["useful"]) == (1, 1, 0)
