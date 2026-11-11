"""Run a pose model over a video and save the keypoints as Parquet.

Usage:
    python -m boulder.pose.run data/test/clip.mp4                    # MediaPipe
    python -m boulder.pose.run data/test/clip.mp4 --model rtmpose
    python -m boulder.pose.run data/test/clip.mp4 --model both       # compare
    python -m boulder.pose.run data/test/*.mp4 --model both --stride 2
    python -m boulder.pose.run data/test/clip.mp4 --model rtmpose --skeleton body_with_feet

Output layout (one file per video per model, "Hive-partitioned" by model):
    data/processed/keypoints/model=mediapipe/<video_id>.parquet
    data/processed/keypoints/model=rtmpose/<video_id>.parquet        (17 joints)
    data/processed/keypoints/model=rtmpose-feet/<video_id>.parquet   (26 joints, with toes/heels)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import pandas as pd

from .base import SCHEMA_COLUMNS, PoseBackend

DEFAULT_OUT = Path("data/processed/keypoints")


def make_backend(model: str, **kwargs) -> PoseBackend:
    if model == "mediapipe":
        from .mediapipe_backend import MediaPipeBackend

        return MediaPipeBackend(**kwargs)
    if model == "rtmpose":
        from .rtmpose_backend import RTMPoseBackend

        return RTMPoseBackend(**kwargs)
    raise ValueError(f"unknown model {model!r}")


def extract_keypoints(video_path: Path, backend: PoseBackend, stride: int = 1) -> pd.DataFrame:
    """Read a video frame by frame and return one row per frame per keypoint."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    video_id = Path(video_path).stem
    rows, frame_idx = [], 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                h, w = frame.shape[:2]
                ts = int(round(frame_idx * 1000 / fps))
                for kp in backend.process_frame(frame, ts):
                    rows.append((
                        video_id, backend.name, backend.version, frame_idx, ts, kp.name,
                        kp.x_px, kp.y_px, kp.x_px / w, kp.y_px / h, kp.confidence, w, h,
                    ))
            frame_idx += 1
    finally:
        cap.release()
    return pd.DataFrame(rows, columns=SCHEMA_COLUMNS)


def output_label(model: str, skeleton: str = "body") -> str:
    """Folder name for a model run: rtmpose with feet gets its own folder."""
    if model == "rtmpose" and skeleton == "body_with_feet":
        return "rtmpose-feet"
    return model


def run(video_path: Path, model: str, out_dir: Path = DEFAULT_OUT, stride: int = 1,
        skeleton: str = "body", rtm_mode: str = "balanced") -> Path:
    start = time.perf_counter()
    kwargs = {"skeleton": skeleton, "mode": rtm_mode} if model == "rtmpose" else {}
    with make_backend(model, **kwargs) as backend:
        df = extract_keypoints(video_path, backend, stride=stride)
    label = output_label(model, skeleton)
    out_path = Path(out_dir) / f"model={label}" / f"{Path(video_path).stem}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    frames = df["frame_idx"].nunique()
    secs = time.perf_counter() - start
    print(f"[{label}] {Path(video_path).name}: {frames} frames with a person, "
          f"{len(df)} rows, {secs:.1f}s -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--model", choices=["mediapipe", "rtmpose", "both"], default="mediapipe")
    parser.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    parser.add_argument("--skeleton", choices=["body", "body_with_feet"], default="body",
                        help="RTMPose only: 17 joints, or 26 incl. toes and heels")
    parser.add_argument("--rtm-mode", choices=["lightweight", "balanced", "performance"],
                        default="balanced", help="RTMPose only: speed vs accuracy")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    models = ["mediapipe", "rtmpose"] if args.model == "both" else [args.model]
    for video in args.videos:
        for model in models:
            run(video, model, args.out, args.stride, args.skeleton, args.rtm_mode)


if __name__ == "__main__":
    main()
