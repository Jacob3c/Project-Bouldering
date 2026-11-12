"""Profile pose keypoints before writing quality rules: what actually goes wrong?

Read-only: prints a report, writes nothing. Run it, look at the numbers, and
use them to choose thresholds for the quality checks.

Usage:
    python -m boulder.quality.profile                     # rtmpose-feet, all cameras
    python -m boulder.quality.profile --model mediapipe
    python -m boulder.quality.profile --camera tripod

Per clip it reports:
    frames      processed frames with a detected person
    stride      frame step between processed frames (pose --stride)
    missing     processed frames with no person detected (gaps in frame_idx)
    longest_gap longest run of missing frames
    torso_px    median torso length (shoulder midpoint to hip midpoint) in pixels
    out_frame   share of keypoints placed outside the image

Per joint (all clips together):
    conf_ok     share of frames with confidence >= --conf
    jumps       moves larger than --jump torso lengths between two consecutive frames
    p99_step    99th-percentile step size (TL): normal movement sits below this
    max_step    largest single step (TL)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEYPOINT_ROOT = Path("data/processed/keypoints")

KEY_JOINTS = [
    "left_wrist", "right_wrist", "left_elbow", "right_elbow",
    "left_shoulder", "right_shoulder", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_big_toe", "right_big_toe",
]


def torso_length_px(kp: pd.DataFrame) -> float:
    """Median distance from shoulder midpoint to hip midpoint, over all frames."""
    wide = kp.pivot_table(index="frame_idx", columns="keypoint", values=["x_px", "y_px"])

    def mid(axis: str, a: str, b: str) -> pd.Series:
        return (wide[(axis, a)] + wide[(axis, b)]) / 2

    dx = mid("x_px", "left_shoulder", "right_shoulder") - mid("x_px", "left_hip", "right_hip")
    dy = mid("y_px", "left_shoulder", "right_shoulder") - mid("y_px", "left_hip", "right_hip")
    return float(np.nanmedian(np.hypot(dx, dy)))


def joint_steps(kp: pd.DataFrame, conf: float, torso_px: float, stride: int) -> pd.DataFrame:
    """One row per joint per frame: how far the joint moved since the previous processed frame.

    Only confident points, and only steps between consecutive processed frames
    (a step across a gap would look like a jump but is just missing data).
    """
    pts = kp[kp["keypoint"].isin(KEY_JOINTS) & (kp["confidence"] >= conf)]
    pts = pts.sort_values(["keypoint", "frame_idx"])
    grp = pts.groupby("keypoint")
    step_tl = np.hypot(grp["x_px"].diff(), grp["y_px"].diff()) / torso_px
    consecutive = grp["frame_idx"].diff() == stride
    out = pts.loc[consecutive, ["keypoint", "frame_idx"]].copy()
    out["step_tl"] = step_tl[consecutive]
    return out


def profile_clip(kp: pd.DataFrame, conf: float = 0.5) -> tuple[dict, pd.DataFrame]:
    """Summary numbers for one clip, plus its per-joint steps."""
    frames = np.sort(kp["frame_idx"].unique())
    stride = int(np.median(np.diff(frames))) if len(frames) > 1 else 1
    gaps = np.diff(frames) // stride - 1 if len(frames) > 1 else np.array([0])
    torso_px = torso_length_px(kp)
    outside = ((kp["x_norm"] < 0) | (kp["x_norm"] > 1) |
               (kp["y_norm"] < 0) | (kp["y_norm"] > 1))
    summary = {
        "frames": len(frames),
        "stride": stride,
        "missing": int(gaps[gaps > 0].sum()),
        "longest_gap": int(gaps.max()),
        "torso_px": round(torso_px, 1),
        "out_frame": round(float(outside.mean()), 4),
    }
    return summary, joint_steps(kp, conf, torso_px, stride)


def joint_table(kp_all: pd.DataFrame, steps_all: pd.DataFrame, conf: float,
                jump_tl: float) -> pd.DataFrame:
    """Per-joint confidence and step statistics across every clip."""
    sub = kp_all[kp_all["keypoint"].isin(KEY_JOINTS)]
    conf_ok = sub.assign(ok=sub["confidence"] >= conf).groupby("keypoint")["ok"].mean()
    s = steps_all.groupby("keypoint")["step_tl"]
    table = pd.DataFrame({
        "conf_ok": conf_ok,
        "jumps": s.apply(lambda x: int((x > jump_tl).sum())),
        "p99_step": s.quantile(0.99),
        "max_step": s.max(),
    })
    return table.reindex([j for j in KEY_JOINTS if j in table.index]).round(3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="rtmpose-feet")
    parser.add_argument("--camera", default="*", help="handheld | tripod | * (all)")
    parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    parser.add_argument("--jump", type=float, default=0.5,
                        help="step larger than this many torso lengths counts as a jump")
    parser.add_argument("--keypoints", type=Path, default=KEYPOINT_ROOT)
    args = parser.parse_args()

    files = sorted(args.keypoints.glob(f"camera={args.camera}/model={args.model}/*.parquet"))
    if not files:
        raise SystemExit(f"no keypoint files for model={args.model} camera={args.camera}")

    summaries, all_kp, all_steps = [], [], []
    for f in files:
        kp = pd.read_parquet(f)
        summary, steps = profile_clip(kp, args.conf)
        summaries.append({"clip": f.stem, **summary})
        all_kp.append(kp[["keypoint", "confidence"]])
        all_steps.append(steps.assign(clip=f.stem))

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(f"=== clips (model={args.model}, conf>={args.conf}) ===")
    print(pd.DataFrame(summaries).set_index("clip").to_string())

    steps_all = pd.concat(all_steps, ignore_index=True)
    print(f"\n=== joints (jump = step > {args.jump} TL between consecutive frames) ===")
    print(joint_table(pd.concat(all_kp), steps_all, args.conf, args.jump).to_string())

    print("\n=== 10 largest steps (check these frames in the video) ===")
    worst = steps_all.nlargest(10, "step_tl")[["clip", "frame_idx", "keypoint", "step_tl"]]
    print(worst.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
