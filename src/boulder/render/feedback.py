"""Feedback video: the climber's skeleton, with inefficient moments drawn in red.

Usage:
    python -m boulder.render.feedback data/raw/clip.mov
    python -m boulder.render.feedback data/raw/clip.mov --scale 1.0 --model rtmpose-feet

Reads (run these first):
    data/processed/keypoints/camera=<c>/model=<m>/<video_id>.parquet   boulder.pose.run
    data/processed/quality/points/...           (optional) boulder.quality.run
    data/processed/metrics/frames/...           boulder.transforms.run
    data/processed/metrics/attempts/...         (for the MetricConfig used)
Writes:
    data/output/<video_id>_feedback.mp4
    data/processed/feedback/camera=<c>/model=<m>/<video_id>.parquet   one row per flagged event

What turns red (rules are first guesses, thresholds in FeedbackConfig):
    bent arm      elbow below straight_arm_deg for at least bent_arm_min_s in a row.
                  Hanging on bent arms tires you out; straight arms rest on the skeleton.
    foot readjust a foot move shorter than readjust_max_dist (a shuffle, not a step).
                  The foot stays red for flag_hold_s after the move so it can be seen.
    pause         hips still for pause_min_s (orange body). Tripod clips only: with a
                  moving camera hip speed is meaningless.

Outside the climb window (attempts log) the skeleton is grey. A timeline at the
bottom shows every flagged moment, with a white cursor for the current frame.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from boulder.ingest.clip_name import parse_clip_name
from boulder.transforms.metrics import (
    MetricConfig,
    _runs,
    foot_move_spans,
    pause_spans,
)

from .overlay import BONES, HOLD_FRAMES, Pose, _open_writer, poses_by_frame

KEYPOINT_ROOT = Path("data/processed/keypoints")
QUALITY_ROOT = Path("data/processed/quality/points")
METRIC_ROOT = Path("data/processed/metrics")
FEEDBACK_ROOT = Path("data/processed/feedback")
OUTPUT_DIR = Path("data/output")

# BGR colours
OK = (90, 220, 90)          # green: inside the climb window, nothing flagged
OUTSIDE = (170, 170, 170)   # grey: before / after the climb
BAD = (40, 40, 235)         # red: flagged limb
PAUSE = (0, 165, 255)       # orange: whole body during a pause

FLAGS = ["left_arm_bent", "right_arm_bent", "left_foot_readjust", "right_foot_readjust",
         "pause"]
LABELS = {
    "left_arm_bent": "Left arm bent", "right_arm_bent": "Right arm bent",
    "left_foot_readjust": "Left foot readjust", "right_foot_readjust": "Right foot readjust",
    "pause": "Pause",
}


@dataclass
class FeedbackConfig:
    bent_arm_min_s: float = 1.5   # elbow bent at least this long in a row = flagged
    flag_hold_s: float = 0.5      # keep a short flag (foot readjust) visible at least this long

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- which frames are flagged

def _side_of(joint: str) -> str | None:
    return "left" if joint.startswith("left_") else "right" if joint.startswith("right_") else None


def bone_group(a: str, b: str) -> str | None:
    """'left_arm', 'right_leg', ... for a bone, or None for torso bones."""
    side = _side_of(a)
    if side is None or side != _side_of(b):
        return None
    joints = {a.split("_", 1)[1], b.split("_", 1)[1]}
    if joints & {"elbow", "wrist"}:
        return f"{side}_arm"
    if joints & {"ankle", "heel", "big_toe", "small_toe", "foot_index"}:
        return f"{side}_leg"
    return None


def _extend(mask: np.ndarray, t: np.ndarray, min_s: float) -> np.ndarray:
    """Lengthen each True run so it lasts at least min_s (short flags stay visible)."""
    out = mask.copy()
    for start, end in _runs(mask):
        stop = np.searchsorted(t, t[start] + min_s, side="right")
        out[start:max(end, stop)] = True
    return out


def feedback_flags(frames: pd.DataFrame, cfg: MetricConfig | None = None,
                   fcfg: FeedbackConfig | None = None, fixed_camera: bool = False
                   ) -> pd.DataFrame:
    """One row per metric frame: frame_idx, t_s and a True/False column per flag."""
    cfg, fcfg = cfg or MetricConfig(), fcfg or FeedbackConfig()
    out = pd.DataFrame({"frame_idx": frames["frame_idx"].to_numpy(),
                        "t_s": frames["t_s"].to_numpy()})
    t = out["t_s"].to_numpy()
    n = len(out)

    for side in ("left", "right"):
        bent = (frames[f"{side}_elbow_deg"] < cfg.straight_arm_deg).to_numpy()  # NaN -> False
        flag = np.zeros(n, dtype=bool)
        for start, end in _runs(bent):
            if t[end - 1] - t[start] >= fcfg.bent_arm_min_s:
                flag[start:end] = True
        out[f"{side}_arm_bent"] = flag

        flag = np.zeros(n, dtype=bool)
        for start, end, readjust in foot_move_spans(frames, side, cfg):
            if readjust:
                flag[start:end] = True
        out[f"{side}_foot_readjust"] = _extend(flag, t, fcfg.flag_hold_s)

    flag = np.zeros(n, dtype=bool)
    if fixed_camera:
        for start, end in pause_spans(frames, cfg):
            flag[start:end] = True
    out["pause"] = flag
    return out


def flag_events(flags: pd.DataFrame) -> pd.DataFrame:
    """One row per flagged stretch: flag, start_s, end_s, duration_s."""
    rows = []
    t = flags["t_s"].to_numpy()
    for name in FLAGS:
        for start, end in _runs(flags[name].to_numpy()):
            rows.append({"flag": name, "start_s": float(t[start]), "end_s": float(t[end - 1]),
                         "duration_s": float(t[end - 1] - t[start])})
    return pd.DataFrame(rows, columns=["flag", "start_s", "end_s", "duration_s"]
                        ).sort_values("start_s", ignore_index=True)


# ---------------------------------------------------------------- drawing

def draw_feedback_pose(img: np.ndarray, pose: Pose, active: set[str], in_window: bool,
                       conf: float, scale: float = 1.0) -> None:
    """Draw one skeleton; limbs named in `active` flags turn red."""
    pts = {name: (int(round(x * scale)), int(round(y * scale)))
           for name, (x, y, c) in pose.items() if c >= conf}
    if not in_window:
        base = OUTSIDE
    elif "pause" in active:
        base = PAUSE
    else:
        base = OK
    red_groups = {f.rsplit("_", 1)[0].replace("_foot", "_leg") for f in active if f != "pause"}
    # left_arm_bent -> left_arm, left_foot_readjust -> left_leg
    thickness = max(2, int(round(6 * scale)))
    joint_colour: dict[str, tuple[int, int, int]] = {}
    for a, b in BONES:
        if a in pts and b in pts:
            colour = BAD if in_window and bone_group(a, b) in red_groups else base
            cv2.line(img, pts[a], pts[b], colour, thickness, cv2.LINE_AA)
            for j in (a, b):
                if joint_colour.get(j) != BAD:
                    joint_colour[j] = colour
    for name, colour in joint_colour.items():
        cv2.circle(img, pts[name], thickness + 1, colour, -1, cv2.LINE_AA)


def _text(img: np.ndarray, text: str, org: tuple[int, int], colour, size: float) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, size, colour, 2, cv2.LINE_AA)


def draw_labels(img: np.ndarray, active: set[str]) -> None:
    for i, name in enumerate(f for f in FLAGS if f in active):
        _text(img, LABELS[name], (12, 32 + i * 30), PAUSE if name == "pause" else BAD, 0.8)


def timeline_strip(n_video_frames: int, width: int, flagged_frames: np.ndarray,
                   window: tuple[int, int] | None) -> np.ndarray:
    """Background bar for the timeline: dark grey, climb window lighter, flags red."""
    h = 14
    bar = np.full((h, width, 3), 50, dtype=np.uint8)
    if n_video_frames <= 0:
        return bar

    def x(f: float) -> int:
        return int(np.clip(f / n_video_frames * width, 0, width - 1))

    if window is not None:
        bar[:, x(window[0]):x(window[1]) + 1] = 95
    for f in flagged_frames:
        bar[:, x(f)] = BAD
    return bar


# ---------------------------------------------------------------- render

def _lookup(table: dict[int, object], frame_idx: int, last: list) -> object | None:
    """Value for this video frame; frames skipped by --stride reuse the last one."""
    value = table.get(frame_idx)
    if value is not None:
        last[:] = [frame_idx, value]
        return value
    if last and frame_idx - last[0] <= HOLD_FRAMES:
        return last[1]
    return None


def render_feedback(video_path: Path, poses: dict[int, Pose], flags: pd.DataFrame,
                    out_path: Path, conf: float = 0.5, scale: float = 0.5) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    active_by_frame = {int(r.frame_idx): {f for f in FLAGS if getattr(r, f)}
                       for r in flags.itertuples(index=False)}
    window = (int(flags["frame_idx"].min()), int(flags["frame_idx"].max())) if len(flags) else None
    flagged = flags.loc[flags[FLAGS].any(axis=1), "frame_idx"].to_numpy()

    writer, strip = None, None
    last_pose: list = []
    last_flags: list = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if scale != 1.0:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            h, w = frame.shape[:2]
            if writer is None:
                writer = _open_writer(out_path, fps, (w, h))
                strip = timeline_strip(n_video, w, flagged, window)

            in_window = window is not None and window[0] <= frame_idx <= window[1]
            active = _lookup(active_by_frame, frame_idx, last_flags) if in_window else None
            active = active or set()
            pose = _lookup(poses, frame_idx, last_pose)
            if pose:
                draw_feedback_pose(frame, pose, active, in_window, conf, scale)
            draw_labels(frame, active)

            frame[h - strip.shape[0]:, :] = strip
            cx = int(np.clip(frame_idx / n_video * w, 0, w - 1))
            cv2.line(frame, (cx, h - strip.shape[0]), (cx, h - 1), (255, 255, 255), 2)
            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    return out_path


# ---------------------------------------------------------------- command line

def _find(root: Path, model: str, stem: str) -> Path | None:
    matches = sorted(root.glob(f"camera=*/model={model}/{stem}.parquet"))
    return matches[0] if matches else None


def _saved_config(attempt_path: Path | None) -> MetricConfig:
    """The MetricConfig the metrics were computed with, so flags match the numbers."""
    if attempt_path is None:
        return MetricConfig()
    row = pd.read_parquet(attempt_path).iloc[0]
    if "metric_config" not in row or not isinstance(row["metric_config"], str):
        return MetricConfig()
    known = MetricConfig().to_dict()
    saved = {k: v for k, v in json.loads(row["metric_config"]).items() if k in known}
    return MetricConfig(**saved)


def _fmt(s: float) -> str:
    return f"{int(s // 60)}:{s % 60:04.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", default="rtmpose-feet")
    parser.add_argument("--conf", type=float, default=0.5, help="hide joints below this confidence")
    parser.add_argument("--scale", type=float, default=0.5, help="resize output (0.5 = half size)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    stem = args.video.stem
    kp_path = _find(KEYPOINT_ROOT, args.model, stem)
    frames_path = _find(METRIC_ROOT / "frames", args.model, stem)
    if kp_path is None:
        parser.error(f"no keypoints for {stem} (run boulder.pose.run first)")
    if frames_path is None:
        parser.error(f"no metrics for {stem} (run boulder.transforms.run first)")

    keypoints = pd.read_parquet(kp_path)
    quality_path = _find(QUALITY_ROOT, args.model, stem)
    if quality_path is not None:  # hide points the quality checks rejected
        from boulder.transforms.prepare import apply_point_quality
        keypoints, _ = apply_point_quality(keypoints, pd.read_parquet(quality_path))

    camera = parse_clip_name(args.video).camera
    cfg = _saved_config(_find(METRIC_ROOT / "attempts", args.model, stem))
    frames = pd.read_parquet(frames_path)
    # frames t_s starts at 0 at the window start; shift back to clip time for printing
    flags = feedback_flags(frames, cfg, fixed_camera=camera == "tripod")
    clip_t0 = float(frames["timestamp_ms"].iloc[0]) / 1000 if len(frames) else 0.0
    events = flag_events(flags)
    events[["start_s", "end_s"]] += clip_t0

    rel = kp_path.relative_to(KEYPOINT_ROOT)
    events_out = FEEDBACK_ROOT / rel
    events_out.parent.mkdir(parents=True, exist_ok=True)
    events.insert(0, "video_id", stem)
    events.to_parquet(events_out, index=False)

    out = args.out or OUTPUT_DIR / f"{stem}_feedback.mp4"
    render_feedback(args.video, poses_by_frame(keypoints), flags, out, args.conf, args.scale)

    print(f"wrote {out}")
    print(f"wrote {events_out} ({len(events)} events)")
    for e in events.itertuples(index=False):
        print(f"  {_fmt(e.start_s)}  {LABELS[e.flag]:<20} {e.duration_s:.1f} s")
    if camera != "tripod":
        print("pauses not shown: they need a tripod clip")


if __name__ == "__main__":
    main()
