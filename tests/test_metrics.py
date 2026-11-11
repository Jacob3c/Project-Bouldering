"""Metrics tested on synthetic keypoints with known answers."""

import numpy as np
import pandas as pd
import pytest

from boulder.ingest.clip_name import parse_clip_name
from boulder.transforms.metrics import (
    MetricConfig,
    attempt_metrics,
    frame_metrics,
    joint_angle,
)

FPS = 30


def _skeleton(hip_x=500.0, hip_y=1000.0, elbow_bend=0.0, left_foot=(450.0, 1300.0)):
    """One frame of joints. Torso = 200 px. elbow_bend=0 -> straight arms."""
    sh_y = hip_y - 200
    pts = {
        "left_shoulder": (hip_x - 50, sh_y), "right_shoulder": (hip_x + 50, sh_y),
        "left_hip": (hip_x - 40, hip_y), "right_hip": (hip_x + 40, hip_y),
        "left_knee": (hip_x - 45, hip_y + 150), "right_knee": (hip_x + 45, hip_y + 150),
        "left_ankle": (hip_x - 50, hip_y + 300), "right_ankle": (hip_x + 50, hip_y + 300),
        "right_big_toe": (hip_x + 50, hip_y + 320),
        "left_big_toe": left_foot,
    }
    for side, sx in (("left", hip_x - 50), ("right", hip_x + 50)):
        pts[f"{side}_elbow"] = (sx, sh_y - 120)  # upper arm straight up
        if elbow_bend:  # forearm turned sideways -> 90 degree elbow
            pts[f"{side}_wrist"] = (sx + 120, sh_y - 120)
        else:
            pts[f"{side}_wrist"] = (sx, sh_y - 240)
    return pts


def _keypoints(frames):
    rows = []
    for i, pts in enumerate(frames):
        for name, (x, y) in pts.items():
            rows.append((i, int(round(i * 1000 / FPS)), name, x, y, 0.9, "test-0"))
    return pd.DataFrame(rows, columns=["frame_idx", "timestamp_ms", "keypoint",
                                       "x_px", "y_px", "confidence", "model_version"])


def test_joint_angle():
    a = np.array([[0.0, 1.0], [0.0, 1.0]])
    b = np.zeros((2, 2))
    c = np.array([[1.0, 0.0], [0.0, -1.0]])
    assert np.allclose(joint_angle(a, b, c), [90.0, 180.0])


def test_body_scale_and_straight_arms():
    frames = frame_metrics(_keypoints([_skeleton() for _ in range(30)]))
    assert frames["body_scale_px"].iloc[0] == pytest.approx(200.0)
    assert frames["left_elbow_deg"].median() == pytest.approx(180.0)
    row = attempt_metrics(frames)
    assert row["arms_straight_share"] == pytest.approx(1.0)


def test_bent_arms_detected():
    frames = frame_metrics(_keypoints([_skeleton(elbow_bend=1) for _ in range(30)]))
    assert frames["right_elbow_deg"].median() == pytest.approx(90.0)
    assert attempt_metrics(frames)["arms_straight_share"] == pytest.approx(0.0)


def test_low_confidence_joints_ignored():
    kp = _keypoints([_skeleton() for _ in range(30)])
    kp.loc[kp["keypoint"] == "left_wrist", "confidence"] = 0.1
    frames = frame_metrics(kp)
    assert frames["left_elbow_deg"].isna().all()
    assert frames["right_elbow_deg"].notna().all()


def test_pause_then_climb():
    # 2 s still, then the hips rise 400 px (2 torso lengths) over 1 s
    still = [_skeleton() for _ in range(2 * FPS)]
    climb = [_skeleton(hip_y=1000 - 400 * k / FPS) for k in range(1, FPS + 1)]
    frames = frame_metrics(_keypoints(still + climb))
    row = attempt_metrics(frames)
    assert row["pauses"] == 1
    assert row["pause_total_s"] == pytest.approx(2.0, abs=0.25)
    assert row["hip_path_efficiency"] > 0.95  # straight line up


def test_foot_move_vs_readjustment():
    cfg = MetricConfig()
    seq = [_skeleton() for _ in range(FPS)]                                   # still 1 s
    # small shuffle: 40 px = 0.2 torso lengths, at 0.75 TL/s
    seq += [_skeleton(left_foot=(450, 1300 - 5 * k)) for k in range(1, 9)]
    seq += [_skeleton(left_foot=(450, 1260)) for _ in range(FPS)]             # still 1 s
    # real move: 200 px = 1 torso length
    seq += [_skeleton(left_foot=(450, 1260 - 20 * k)) for k in range(1, 11)]
    seq += [_skeleton(left_foot=(450, 1060)) for _ in range(FPS)]             # still 1 s
    row = attempt_metrics(frame_metrics(_keypoints(seq), cfg), cfg)
    assert row["left_foot_moves"] == 2
    assert row["left_foot_readjustments"] == 1  # only the shuffle
    assert row["right_foot_moves"] == 0


def test_parse_clip_name():
    info = parse_clip_name("data/test/2026-09-12_Stone_CO_Purple-V3_att01_send_handheld.MOV")
    assert info.name_ok
    assert (info.date, info.gym, info.colour, info.grade) == ("2026-09-12", "Stone_CO", "Purple", "V3")
    assert (info.attempt, info.outcome, info.camera) == (1, "send", "handheld")
    assert parse_clip_name("random_clip.mp4").name_ok is False
