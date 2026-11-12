"""Quality checks on pose keypoints: which frames and points can metrics trust?

Three levels, thresholds in QualityConfig (chosen from the profile of the first
5 clips; see docs/decisions.md):

frame  body jump: the hip midpoint lands more than body_jump_tl torso lengths
       from the climber's last good position. Happens when someone walks in
       front of the camera (the pose model then picks the passer-by as the
       largest person) or the camera jerks. The frame stays invalid until the
       skeleton comes back; if it stays away longer than rebase_after_s, the new
       position is accepted (the camera moved for good).
point  low confidence (< min_confidence), or a spike: a joint more than
       joint_jump_tl torso lengths from the rolling median of its neighbouring
       frames, measured relative to the hips so camera shake cancels out.
clip   pass / warn / fail with reasons: climber too small in frame, points out
       of frame, long detection gaps, handheld camera, many invalid frames.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from boulder.quality.profile import torso_length_px

PASS, WARN, FAIL = "pass", "warn", "fail"


@dataclass
class QualityConfig:
    min_confidence: float = 0.5     # points below this are dropped
    body_jump_tl: float = 1.5       # hip this far from last good position = invalid frame
    rebase_after_s: float = 2.0     # accept a new position after this long away
    joint_jump_tl: float = 1.0      # joint this far from its rolling median = spike
    spike_window: int = 5           # frames in the rolling median (odd number)
    min_torso_px: float = 100.0     # smaller climber in frame -> noisy metrics
    max_out_of_frame: float = 0.01  # share of points outside the image
    max_gap_s: float = 0.5          # longest stretch with no person detected
    max_invalid_share: float = 0.2  # more invalid frames than this -> fail

    def to_dict(self) -> dict:
        return asdict(self)


def hip_midpoints(kp: pd.DataFrame, cfg: QualityConfig) -> pd.DataFrame:
    """One row per frame: timestamp and hip midpoint (NaN when a hip is not confident)."""
    frames = kp.groupby("frame_idx")["timestamp_ms"].first().to_frame()
    hips = kp[kp["keypoint"].isin(["left_hip", "right_hip"])
              & (kp["confidence"] >= cfg.min_confidence)]
    wide = hips.pivot_table(index="frame_idx", columns="keypoint", values=["x_px", "y_px"])
    for axis in ("x", "y"):
        cols = [(f"{axis}_px", "left_hip"), (f"{axis}_px", "right_hip")]
        if all(c in wide.columns for c in cols):
            frames[f"hip_{axis}"] = (wide[cols[0]] + wide[cols[1]]) / 2
        else:
            frames[f"hip_{axis}"] = np.nan
    return frames


def mark_body_jumps(frames: pd.DataFrame, torso_px: float, cfg: QualityConfig) -> pd.DataFrame:
    """Add body_dist_tl (distance from last good hip position) and valid (bool) per frame."""
    hx, hy = frames["hip_x"].to_numpy(), frames["hip_y"].to_numpy()
    ts = frames["timestamp_ms"].to_numpy(dtype=float)
    valid = np.ones(len(frames), dtype=bool)
    dist = np.full(len(frames), np.nan)
    anchor = None        # last good (x, y)
    away_since = None    # timestamp the skeleton first left the anchor

    for i in range(len(frames)):
        if np.isnan(hx[i]) or np.isnan(hy[i]):
            continue                       # no hip: can't judge, leave valid
        if anchor is None:
            anchor = (hx[i], hy[i])
            continue
        dist[i] = np.hypot(hx[i] - anchor[0], hy[i] - anchor[1]) / torso_px
        if dist[i] <= cfg.body_jump_tl:
            anchor, away_since = (hx[i], hy[i]), None
            continue
        away_since = ts[i] if away_since is None else away_since
        if (ts[i] - away_since) / 1000 >= cfg.rebase_after_s:
            anchor, away_since = (hx[i], hy[i]), None   # camera moved for good
        else:
            valid[i] = False

    return frames.assign(body_dist_tl=dist, valid=valid)


def mark_points(kp: pd.DataFrame, frames: pd.DataFrame, torso_px: float,
                cfg: QualityConfig) -> pd.DataFrame:
    """Return keypoints with point_ok (bool) and the frame's valid flag added."""
    pts = kp.merge(frames[["hip_x", "hip_y", "valid"]], left_on="frame_idx",
                   right_index=True, how="left").sort_values(["keypoint", "frame_idx"])
    confident = pts["confidence"] >= cfg.min_confidence
    pts["_rx"] = (pts["x_px"] - pts["hip_x"]).where(confident & pts["valid"])
    pts["_ry"] = (pts["y_px"] - pts["hip_y"]).where(confident & pts["valid"])

    def rolling_median(s: pd.Series) -> pd.Series:
        return s.rolling(cfg.spike_window, center=True, min_periods=3).median()

    med = pts.groupby("keypoint")[["_rx", "_ry"]].transform(rolling_median)
    deviation = np.hypot(pts["_rx"] - med["_rx"], pts["_ry"] - med["_ry"]) / torso_px
    spike = (deviation > cfg.joint_jump_tl).fillna(False)

    pts["point_ok"] = confident & pts["valid"] & ~spike
    return pts.drop(columns=["_rx", "_ry", "hip_x", "hip_y"]).sort_values(
        ["frame_idx", "keypoint"]).reset_index(drop=True)


def clip_report(points: pd.DataFrame, frames: pd.DataFrame, torso_px: float, camera: str,
                cfg: QualityConfig) -> dict:
    """One-row summary: numbers, status (pass/warn/fail) and the reasons behind it."""
    ts = frames["timestamp_ms"].to_numpy(dtype=float)
    steps = np.diff(ts)
    longest_gap_s = float((steps.max() - np.median(steps)) / 1000) if len(steps) else 0.0
    outside = ((points["x_norm"] < 0) | (points["x_norm"] > 1) |
               (points["y_norm"] < 0) | (points["y_norm"] > 1)).mean()
    invalid_share = float(1 - frames["valid"].mean())

    fails, warns = [], []
    if invalid_share > cfg.max_invalid_share:
        fails.append(f"{invalid_share:.0%} of frames invalid")
    elif invalid_share > 0:
        warns.append(f"{(~frames['valid']).sum()} invalid frames (body jumps)")
    if torso_px < cfg.min_torso_px:
        warns.append(f"climber small in frame (torso {torso_px:.0f} px < {cfg.min_torso_px:.0f})")
    if outside > cfg.max_out_of_frame:
        warns.append(f"{outside:.1%} of points out of frame")
    if longest_gap_s > cfg.max_gap_s:
        warns.append(f"detection gap of {longest_gap_s:.1f} s")
    if camera == "handheld":
        warns.append("handheld camera: camera motion not compensated")

    status = FAIL if fails else WARN if warns else PASS
    return {
        "status": status,
        "reasons": "; ".join(fails + warns),
        "frames": len(frames),
        "invalid_frames": int((~frames["valid"]).sum()),
        "dropped_points_share": round(float(1 - points["point_ok"].mean()), 4),
        "torso_px": round(torso_px, 1),
        "out_of_frame_share": round(float(outside), 4),
        "longest_gap_s": round(longest_gap_s, 2),
    }


def check_clip(kp: pd.DataFrame, camera: str, cfg: QualityConfig | None = None
               ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run every check on one clip's keypoints: (points, frames, report)."""
    cfg = cfg or QualityConfig()
    torso_px = torso_length_px(kp)
    frames = mark_body_jumps(hip_midpoints(kp, cfg), torso_px, cfg)
    points = mark_points(kp, frames, torso_px, cfg)
    return points, frames, clip_report(points, frames, torso_px, camera, cfg)
