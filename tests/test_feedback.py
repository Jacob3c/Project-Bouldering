"""Feedback flags and renderer, tested on synthetic skeletons."""

import math

import cv2
import numpy as np
import pandas as pd

from boulder.render.feedback import (
    BAD,
    OK,
    bone_group,
    draw_feedback_pose,
    feedback_flags,
    flag_events,
    render_feedback,
)
from boulder.render.overlay import poses_by_frame
from boulder.transforms.metrics import frame_metrics

FPS = 30


def _skeleton(hip_y=1000.0, bent=False, left_foot=(450.0, 1300.0), elbow_deg=None):
    """Torso 200 px; arms straight up unless bent (90 degree elbows) or elbow_deg given."""
    hx, sh_y = 500.0, hip_y - 200
    pts = {
        "left_shoulder": (hx - 50, sh_y), "right_shoulder": (hx + 50, sh_y),
        "left_hip": (hx - 40, hip_y), "right_hip": (hx + 40, hip_y),
        "left_knee": (hx - 45, hip_y + 150), "right_knee": (hx + 45, hip_y + 150),
        "left_ankle": (hx - 50, hip_y + 300), "right_ankle": (hx + 50, hip_y + 300),
        "right_big_toe": (hx + 50, hip_y + 320), "left_big_toe": left_foot,
    }
    for side, sx in (("left", hx - 50), ("right", hx + 50)):
        pts[f"{side}_elbow"] = (sx, sh_y - 120)
        deg = elbow_deg if elbow_deg is not None else 90.0 if bent else 180.0
        a = math.radians(deg)  # angle between forearm and the upper arm's line down to the shoulder
        pts[f"{side}_wrist"] = (sx + 120 * math.sin(a), sh_y - 120 + 120 * math.cos(a))
    return pts


def _keypoints(frames):
    rows = [(i, int(round(i * 1000 / FPS)), name, x, y, 0.9, "test-0")
            for i, pts in enumerate(frames) for name, (x, y) in pts.items()]
    return pd.DataFrame(rows, columns=["frame_idx", "timestamp_ms", "keypoint",
                                       "x_px", "y_px", "confidence", "model_version"])


def _flags(seq, **kw):
    return feedback_flags(frame_metrics(_keypoints(seq)), **kw)


def test_long_bent_arm_flagged_short_one_not():
    seq = [_skeleton() for _ in range(FPS)]
    seq += [_skeleton(bent=True) for _ in range(FPS)]        # 1 s bent: allowed
    seq += [_skeleton() for _ in range(FPS)]
    seq += [_skeleton(bent=True) for _ in range(3 * FPS)]    # 3 s bent: flagged
    seq += [_skeleton() for _ in range(FPS)]
    flags = _flags(seq)
    assert not flags["left_arm_bent"].iloc[FPS:2 * FPS].any()
    assert flags["left_arm_bent"].iloc[3 * FPS + 10:6 * FPS - 10].all()
    events = flag_events(flags)
    assert (events["flag"] == "left_arm_bent").sum() == 1


def test_slightly_bent_arm_not_flagged():
    flags = _flags([_skeleton(elbow_deg=135) for _ in range(4 * FPS)])  # under 150, over 120
    assert not flags["left_arm_bent"].any()
    flags = _flags([_skeleton(elbow_deg=100) for _ in range(4 * FPS)])
    assert flags["left_arm_bent"].all()


def test_readjustment_flagged_real_step_not():
    seq = [_skeleton() for _ in range(FPS)]
    seq += [_skeleton(left_foot=(450, 1300 - 5 * k)) for k in range(1, 9)]       # shuffle
    seq += [_skeleton(left_foot=(450, 1260)) for _ in range(FPS)]
    seq += [_skeleton(left_foot=(450, 1260 - 20 * k)) for k in range(1, 11)]     # real step
    seq += [_skeleton(left_foot=(450, 1060)) for _ in range(FPS)]
    events = flag_events(_flags(seq))
    left = events[events["flag"] == "left_foot_readjust"]
    assert len(left) == 1
    assert left["start_s"].iloc[0] < 1.6          # the shuffle at ~1 s, not the step at ~2.3 s
    assert left["duration_s"].iloc[0] >= 0.45     # held visible for flag_hold_s
    assert not (events["flag"] == "right_foot_readjust").any()


def test_pause_only_with_fixed_camera():
    seq = [_skeleton() for _ in range(2 * FPS)]
    seq += [_skeleton(hip_y=1000 - 400 * k / FPS) for k in range(1, FPS + 1)]
    assert not _flags(seq)["pause"].any()
    assert _flags(seq, fixed_camera=True)["pause"].any()


def test_bone_groups():
    assert bone_group("left_shoulder", "left_elbow") == "left_arm"
    assert bone_group("right_elbow", "right_wrist") == "right_arm"
    assert bone_group("left_knee", "left_ankle") == "left_leg"
    assert bone_group("right_heel", "right_big_toe") == "right_leg"
    assert bone_group("left_hip", "left_knee") is None
    assert bone_group("left_shoulder", "right_shoulder") is None


def test_flagged_arm_drawn_red():
    pose = {k: (x / 10, y / 10, 0.9) for k, (x, y) in _skeleton().items()}
    img = np.zeros((160, 120, 3), dtype=np.uint8)
    draw_feedback_pose(img, pose, {"left_arm_bent"}, in_window=True, conf=0.5)
    # middle of the left forearm (45, 62) and of the right forearm (55, 62)
    assert tuple(img[62, 45]) == BAD
    assert tuple(img[62, 55]) == OK


def test_render_feedback_writes_every_frame(tmp_path):
    video = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (120, 160))
    for _ in range(80):
        writer.write(np.zeros((160, 120, 3), dtype=np.uint8))
    writer.release()

    seq = [_skeleton(bent=True) for _ in range(80)]
    kp = _keypoints(seq)
    kp[["x_px", "y_px"]] /= 10
    flags = feedback_flags(frame_metrics(kp))
    out = render_feedback(video, poses_by_frame(kp), flags, tmp_path / "out.mp4", scale=1.0)
    cap = cv2.VideoCapture(str(out))
    n = 0
    while cap.read()[0]:
        n += 1
    assert n == 80
