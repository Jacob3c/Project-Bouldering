"""Overlay renderer and runner helpers, tested without real models."""

import cv2
import numpy as np
import pandas as pd

from boulder.pose.run import output_label
from boulder.render.overlay import draw_pose, poses_by_frame, render


def _make_video(path, n_frames=5, w=80, h=60, fps=30):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    for _ in range(n_frames):
        writer.write(np.zeros((h, w, 3), dtype=np.uint8))
    writer.release()


def _rows(frames):
    return pd.DataFrame(
        [(f, name, x, y, 0.9)
         for f in frames
         for name, x, y in [("left_shoulder", 20, 10), ("left_elbow", 20, 30)]],
        columns=["frame_idx", "keypoint", "x_px", "y_px", "confidence"],
    )


def test_output_label():
    assert output_label("mediapipe") == "mediapipe"
    assert output_label("rtmpose") == "rtmpose"
    assert output_label("rtmpose", "body_with_feet") == "rtmpose-feet"


def test_poses_by_frame_groups_rows():
    poses = poses_by_frame(_rows([0, 2]))
    assert sorted(poses) == [0, 2]
    assert poses[0]["left_elbow"] == (20.0, 30.0, 0.9)


def test_draw_pose_skips_low_confidence():
    img = np.zeros((60, 80, 3), dtype=np.uint8)
    draw_pose(img, {"left_shoulder": (20, 10, 0.1), "left_elbow": (20, 30, 0.1)},
              (255, 255, 255), conf=0.5)
    assert img.sum() == 0
    draw_pose(img, {"left_shoulder": (20, 10, 0.9), "left_elbow": (20, 30, 0.9)},
              (255, 255, 255), conf=0.5)
    assert img[20, 20].sum() > 0  # bone drawn between the two joints


def test_render_writes_every_frame(tmp_path):
    video = tmp_path / "clip.avi"
    _make_video(video)
    out = render(video, {"fake": poses_by_frame(_rows([0, 2, 4]))}, tmp_path / "out.mp4",
                 scale=1.0)
    cap = cv2.VideoCapture(str(out))
    n = 0
    while cap.read()[0]:
        n += 1
    assert n == 5
