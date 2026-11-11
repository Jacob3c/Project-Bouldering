"""Draw pose skeletons from one or more models on top of the original video.

Usage:
    python -m boulder.render.overlay data/test/clip.MOV
    python -m boulder.render.overlay data/test/clip.MOV --models mediapipe rtmpose-feet
    python -m boulder.render.overlay data/test/clip.MOV --conf 0.3 --scale 0.5

Reads keypoints from data/processed/keypoints/model=<name>/<video_id>.parquet
(run boulder.pose.run first) and writes data/output/<video_id>_overlay.mp4.

Each model gets its own colour. Joints below --conf are not drawn, so gaps in
a skeleton show exactly where that model lost track of you.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

KEYPOINT_ROOT = Path("data/processed/keypoints")
OUTPUT_DIR = Path("data/output")

# Bones drawn between named joints. A bone is skipped when either end is
# missing, so one list works for MediaPipe (33), RTMPose (17) and RTMPose-feet (26).
BONES = [
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    # feet: MediaPipe names
    ("left_ankle", "left_heel"),
    ("left_heel", "left_foot_index"),
    ("left_ankle", "left_foot_index"),
    ("right_ankle", "right_heel"),
    ("right_heel", "right_foot_index"),
    ("right_ankle", "right_foot_index"),
    # feet: RTMPose-feet (Halpe-26) names
    ("left_heel", "left_big_toe"),
    ("left_ankle", "left_big_toe"),
    ("left_big_toe", "left_small_toe"),
    ("right_heel", "right_big_toe"),
    ("right_ankle", "right_big_toe"),
    ("right_big_toe", "right_small_toe"),
]

# BGR colours, one per model slot
COLOURS = [(0, 200, 255), (255, 120, 0), (80, 220, 80), (200, 80, 255)]

# How many frames a pose is held when using --stride (frames in between have no data)
HOLD_FRAMES = 3

Pose = dict[str, tuple[float, float, float]]  # joint -> (x_px, y_px, confidence)


def poses_by_frame(df: pd.DataFrame) -> dict[int, Pose]:
    """Turn keypoint rows into {frame_idx: {joint: (x, y, conf)}}."""
    out: dict[int, Pose] = {}
    for frame_idx, x, y, conf, name in zip(
        df["frame_idx"], df["x_px"], df["y_px"], df["confidence"], df["keypoint"]
    ):
        out.setdefault(int(frame_idx), {})[name] = (float(x), float(y), float(conf))
    return out


def draw_pose(img: np.ndarray, pose: Pose, colour: tuple[int, int, int],
              conf: float, scale: float = 1.0) -> None:
    """Draw bones and joints in place. Joints below `conf` are skipped."""
    pts = {
        name: (int(round(x * scale)), int(round(y * scale)))
        for name, (x, y, c) in pose.items()
        if c >= conf
    }
    thickness = max(2, int(round(6 * scale)))
    for a, b in BONES:
        if a in pts and b in pts:
            cv2.line(img, pts[a], pts[b], colour, thickness, cv2.LINE_AA)
    radius = thickness + 1
    for name, p in pts.items():
        if any(name in bone for bone in BONES):
            cv2.circle(img, p, radius, colour, -1, cv2.LINE_AA)


def draw_legend(img: np.ndarray, labels: list[str]) -> None:
    for i, label in enumerate(labels):
        y = 30 + i * 30
        cv2.rectangle(img, (10, y - 18), (34, y + 4), COLOURS[i % len(COLOURS)], -1)
        cv2.putText(img, label, (42, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4,
                    cv2.LINE_AA)
        cv2.putText(img, label, (42, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                    cv2.LINE_AA)


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    # avc1 (H.264) plays everywhere; fall back to mp4v if this OpenCV build lacks it
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"could not open a video writer for {path}")


def render(video_path: Path, model_poses: dict[str, dict[int, Pose]], out_path: Path,
           conf: float = 0.5, scale: float = 1.0) -> Path:
    """Write a copy of `video_path` with every model's skeleton drawn on it."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    labels = list(model_poses)
    last: dict[str, tuple[int, Pose]] = {}
    writer = None
    frame_idx = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if scale != 1.0:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if writer is None:
                h, w = frame.shape[:2]
                writer = _open_writer(out_path, fps, (w, h))
            for i, label in enumerate(labels):
                pose = model_poses[label].get(frame_idx)
                if pose is not None:
                    last[label] = (frame_idx, pose)
                elif label in last and frame_idx - last[label][0] <= HOLD_FRAMES:
                    pose = last[label][1]
                if pose:
                    draw_pose(frame, pose, COLOURS[i % len(COLOURS)], conf, scale)
            draw_legend(frame, labels)
            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--models", nargs="+", default=["mediapipe", "rtmpose"],
                        help="model folder names under data/processed/keypoints")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="hide joints below this confidence")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="resize output (0.5 = half size, smaller file)")
    parser.add_argument("--root", type=Path, default=KEYPOINT_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    stem = args.video.stem
    model_poses = {}
    for model in args.models:
        path = args.root / f"model={model}" / f"{stem}.parquet"
        if not path.exists():
            parser.error(f"no keypoints for {model}: {path} (run boulder.pose.run first)")
        model_poses[model] = poses_by_frame(pd.read_parquet(path))

    out = args.out or OUTPUT_DIR / f"{stem}_overlay.mp4"
    render(args.video, model_poses, out, args.conf, args.scale)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
