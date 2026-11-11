"""Pose module tests that need no real model: a fake backend stands in."""

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
