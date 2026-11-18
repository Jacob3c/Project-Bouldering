"""Pose module tests that need no real model: a fake backend stands in."""

from pathlib import Path

import cv2
import numpy as np

from boulder.pose.base import SCHEMA_COLUMNS, Keypoint, PoseBackend
from boulder.pose.rtmpose_backend import pick_main_person
from boulder.pose.run import extract_keypoints


class FakeBackend(PoseBackend):
    name = "fake"
    version = "fake-0"

    def __init__(self):
        self.timestamps = []

    def process_frame(self, frame_bgr, timestamp_ms):
        self.timestamps.append(timestamp_ms)
        h, w = frame_bgr.shape[:2]
        return [Keypoint("left_wrist", w / 2, h / 4, 0.9)]


def _make_video(path, n_frames=6, w=64, h=48, fps=30):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    for _ in range(n_frames):
        writer.write(np.zeros((h, w, 3), dtype=np.uint8))
    writer.release()


def test_extract_keypoints_schema_and_stride(tmp_path):
    video = tmp_path / "clip.avi"
    _make_video(video)
    backend = FakeBackend()
    df = extract_keypoints(video, backend, stride=2)

    assert list(df.columns) == SCHEMA_COLUMNS
    assert df["frame_idx"].tolist() == [0, 2, 4]
    assert backend.timestamps == sorted(backend.timestamps)  # must increase
    assert df["video_id"].unique().tolist() == ["clip"]
    assert np.allclose(df["x_norm"], 0.5) and np.allclose(df["y_norm"], 0.25)


def test_pick_main_person_chooses_largest_skeleton():
    small = np.array([[0, 0], [10, 10]], dtype=float)
    big = np.array([[0, 0], [100, 200]], dtype=float)
    scores = np.ones((2, 2))
    assert pick_main_person(np.stack([small, big]), scores) == 1


def test_pick_main_person_none_when_empty():
    assert pick_main_person(np.empty((0, 17, 2)), np.empty((0, 17))) is None


def test_camera_from_name():
    from boulder.pose.run import camera_from_name

    assert camera_from_name("2026-11-08_Gym_Orange-V5_att01_send_handheld.MOV") == "handheld"
    assert camera_from_name("2026-11-12_Gym_Blue-V3_att01_send_TRIPOD.mov") == "tripod"
    assert camera_from_name("clip.mp4") == "unknown"


def test_output_path_partitions_by_camera_then_model(tmp_path):
    from boulder.pose.run import output_path

    p = output_path("x_send_tripod.MOV", "rtmpose-feet", tmp_path)
    assert p == tmp_path / "camera=tripod" / "model=rtmpose-feet" / "x_send_tripod.parquet"


def test_skip_existing_does_not_load_model(tmp_path, monkeypatch):
    import boulder.pose.run as run_mod

    out = run_mod.output_path("a_handheld.MOV", "mediapipe", tmp_path)
    out.parent.mkdir(parents=True)
    out.touch()
    monkeypatch.setattr(run_mod, "make_backend",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("model loaded")))
    assert run_mod.run(Path("a_handheld.MOV"), "mediapipe", tmp_path, skip_existing=True) is None


# --- climber tracking ---

from boulder.pose.rtmpose_backend import (  # noqa: E402
    HALPE26,
    TRACK_RESET_FRAMES,
    RTMPoseBackend,
    person_center_size,
    pick_tracked_person,
)


def _person(cx, cy, height):
    """A 26-joint skeleton: points spread over a box of the given height, centred on (cx, cy)."""
    ys = np.linspace(cy - height / 2, cy + height / 2, 26)
    xs = cx + np.tile([-height / 8, height / 8], 13)
    return np.stack([xs, ys], axis=1)


def _people(*persons):
    kps = np.stack(persons)
    return kps, np.ones(kps.shape[:2])


def test_person_center_size():
    center, size = person_center_size(*[a[0] for a in _people(_person(100, 200, 300))])
    assert np.allclose(center, [100, 200])
    assert 300 <= size <= 310


def test_tracking_keeps_climber_when_bigger_passerby_appears():
    climber = _person(500, 500, 300)
    passerby = _person(520, 600, 900)          # in front of the climber, 3x bigger
    kps, sc = _people(passerby, climber)
    assert pick_tracked_person(kps, sc, None, None) == 0            # no history: biggest
    assert pick_tracked_person(kps, sc, np.array([500, 500]), 305) == 1


def test_tracking_returns_none_when_only_passerby_visible():
    kps, sc = _people(_person(1500, 500, 300))  # same size but far away
    assert pick_tracked_person(kps, sc, np.array([500, 500]), 305) is None


def test_tracking_follows_normal_movement():
    kps, sc = _people(_person(560, 470, 300), _person(1500, 500, 300))
    assert pick_tracked_person(kps, sc, np.array([500, 500]), 305) == 0


class _FakeModel:
    """Stands in for rtmlib: returns the scripted people for each call."""

    def __init__(self, frames):
        self.frames = iter(frames)

    def __call__(self, frame_bgr):
        return next(self.frames)


def _backend(frames):
    b = RTMPoseBackend.__new__(RTMPoseBackend)  # skip __init__ (no rtmlib needed)
    b._model, b._names, b.skeleton, b.mode = _FakeModel(frames), HALPE26, "body_with_feet", "x"
    b.reset()
    return b


def test_backend_skips_passerby_then_recovers():
    climber, passerby = _person(500, 500, 300), _person(1400, 500, 900)
    script = [_people(climber), _people(passerby), _people(passerby, climber)]
    b = _backend(script)
    img = np.zeros((10, 10, 3), np.uint8)
    first = b.process_frame(img, 0)
    assert np.isclose(np.median([k.x_px for k in first]), 500)
    assert b.process_frame(img, 66) == []              # only the passer-by: no skeleton
    third = b.process_frame(img, 132)
    assert np.isclose(np.median([k.x_px for k in third]), 500)


def test_backend_resets_after_long_absence_and_on_new_video():
    climber, elsewhere = _person(500, 500, 300), _person(1500, 500, 300)
    script = [_people(climber)] + [_people(elsewhere)] * (TRACK_RESET_FRAMES + 1)
    b = _backend(script)
    img = np.zeros((10, 10, 3), np.uint8)
    b.process_frame(img, 0)
    for i in range(TRACK_RESET_FRAMES):
        assert b.process_frame(img, 66 * (i + 1)) == []
    assert b._prev_center is None                     # gave up on the old position
    assert b.process_frame(img, 66 * 40) != []        # biggest person picked again

    b2 = _backend([_people(climber), _people(elsewhere)])
    b2.process_frame(img, 1000)
    assert b2.process_frame(img, 0) != []             # timestamp went back: fresh start
