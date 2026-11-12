"""Quality-check tests on a synthetic skeleton with planted problems."""

import pandas as pd

from boulder.quality.checks import FAIL, PASS, WARN, QualityConfig, check_clip

# Joint offsets from the hip midpoint (pixels). Torso = shoulder mid to hip mid = 100 px.
OFFSETS = {
    "left_shoulder": (-20, -100), "right_shoulder": (20, -100),
    "left_elbow": (-35, -60), "right_elbow": (35, -60),
    "left_wrist": (-40, -130), "right_wrist": (40, -130),
    "left_hip": (-15, 0), "right_hip": (15, 0),
    "left_knee": (-15, 60), "right_knee": (15, 60),
    "left_ankle": (-15, 110), "right_ankle": (15, 110),
    "left_heel": (-20, 115), "right_heel": (20, 115),
    "left_big_toe": (-5, 118), "right_big_toe": (5, 118),
}


def _clip(n_frames=30, shift=None, spikes=None, low_conf=None, scale=1.0):
    """Climber drifting slowly right; stride 2, 33 ms per source frame.

    shift:    {frame_idx: (dx, dy)} moves the whole skeleton (passer-by / camera)
    spikes:   {(frame_idx, joint): (dx, dy)} moves one joint
    low_conf: {(frame_idx, joint)} gets confidence 0.1
    """
    shift, spikes, low_conf = shift or {}, spikes or {}, low_conf or set()
    rows = []
    for f in range(0, 2 * n_frames, 2):
        cx, cy = 300 + f, 500
        dx, dy = shift.get(f, (0, 0))
        for joint, (ox, oy) in OFFSETS.items():
            sx, sy = spikes.get((f, joint), (0, 0))
            x = cx + ox * scale + dx + sx
            y = cy + oy * scale + dy + sy
            conf = 0.1 if (f, joint) in low_conf else 0.9
            rows.append((f, f * 33, joint, x, y, x / 1000, y / 1000, conf))
    return pd.DataFrame(rows, columns=["frame_idx", "timestamp_ms", "keypoint", "x_px",
                                       "y_px", "x_norm", "y_norm", "confidence"])


def _invalid(frames):
    return frames.index[~frames["valid"]].tolist()


def test_clean_clip_passes():
    points, frames, report = check_clip(_clip(), "tripod")
    assert frames["valid"].all() and points["point_ok"].all()
    assert report["status"] == PASS and report["reasons"] == ""


def test_passerby_frames_stay_invalid_until_climber_returns():
    passerby = {f: (400, 0) for f in range(20, 30, 2)}   # frames 20-28 on someone else
    _, frames, report = check_clip(_clip(shift=passerby), "tripod")
    assert _invalid(frames) == [20, 22, 24, 26, 28]
    assert report["status"] == WARN and "5 invalid frames" in report["reasons"]


def test_single_joint_spike_drops_only_that_point():
    points, frames, _ = check_clip(_clip(spikes={(12, "left_wrist"): (200, 0)}), "tripod")
    bad = points[~points["point_ok"]]
    assert list(zip(bad["frame_idx"], bad["keypoint"])) == [(12, "left_wrist")]
    assert frames["valid"].all()


def test_low_confidence_point_is_dropped():
    points, _, _ = check_clip(_clip(low_conf={(8, "right_knee")}), "tripod")
    bad = points[~points["point_ok"]]
    assert list(zip(bad["frame_idx"], bad["keypoint"])) == [(8, "right_knee")]


def test_permanent_camera_shift_is_accepted_after_rebase_time():
    moved = {f: (400, 0) for f in range(20, 60, 2)}       # camera moved for good at frame 20
    cfg = QualityConfig(rebase_after_s=0.2)
    _, frames, _ = check_clip(_clip(shift=moved), "tripod", cfg)
    # frames 20-26 are within 0.2 s of leaving; frame 28 (264 ms later) becomes the new anchor
    assert _invalid(frames) == [20, 22, 24, 26]


def test_clip_warnings_and_fail():
    _, _, report = check_clip(_clip(), "handheld")
    assert report["status"] == WARN and "handheld" in report["reasons"]

    _, _, report = check_clip(_clip(scale=0.5), "tripod")          # torso 50 px
    assert "climber small in frame" in report["reasons"]

    mostly_away = {f: (400, 0) for f in range(10, 60, 2)}           # 25 of 30 frames away
    _, _, report = check_clip(_clip(shift=mostly_away), "tripod", QualityConfig(rebase_after_s=99))
    assert report["status"] == FAIL and "frames invalid" in report["reasons"]
