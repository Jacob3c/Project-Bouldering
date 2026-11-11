"""Turn pose keypoints into climbing metrics.

Two outputs per attempt (clip):

frame metrics (one row per processed frame)
    elbow / knee angles, hip centre, foot positions, speeds, flags
attempt metrics (one row per clip)
    straight-arm share, foot moves and readjustments, pauses,
    hip-path efficiency and smoothness, plus metadata from the file name

All distances are divided by the climber's torso length (shoulder midpoint to
hip midpoint, median over the clip), so clips filmed from different distances
are comparable. Unit: "torso lengths" (TL). Image y points down.

Thresholds live in MetricConfig. They are first guesses, meant to be
calibrated later by comparing sends with falls on tripod footage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class MetricConfig:
    min_confidence: float = 0.5      # joints below this are treated as missing
    max_gap_s: float = 0.25          # fill gaps shorter than this by interpolation
    smooth_window_s: float = 0.2     # centred rolling mean over positions
    straight_arm_deg: float = 150.0  # elbow angle at or above = straight arm
    foot_move_speed: float = 0.6     # TL/s; foot faster than this = moving
    foot_still_s: float = 0.3        # foot must be still this long between moves
    readjust_max_dist: float = 0.3   # TL; a foot move shorter than this = readjustment
    pause_speed: float = 0.15        # TL/s; hip slower than this = paused
    pause_min_s: float = 1.0         # stillness at least this long counts as a pause

    def to_dict(self) -> dict:
        return asdict(self)


# First available name wins: RTMPose-feet, MediaPipe, then the ankle as fallback.
FOOT_POINT = {
    "left": ["left_big_toe", "left_foot_index", "left_ankle"],
    "right": ["right_big_toe", "right_foot_index", "right_ankle"],
}

NEEDED = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


# ---------------------------------------------------------------- helpers

def to_wide(keypoints: pd.DataFrame, min_confidence: float) -> pd.DataFrame:
    """Long keypoint rows -> one row per frame with <joint>_x / <joint>_y columns.

    Joints below `min_confidence` become NaN.
    """
    kp = keypoints.copy()
    low = kp["confidence"] < min_confidence
    kp.loc[low, ["x_px", "y_px"]] = np.nan
    # pivot on frame only: pivoting on (frame, timestamp) with dropna=False would
    # create every frame x timestamp combination
    wide = kp.pivot_table(index="frame_idx", columns="keypoint",
                          values=["x_px", "y_px"], aggfunc="first", dropna=False)
    wide.columns = [f"{joint}_{axis[0]}" for axis, joint in wide.columns]
    times = kp.groupby("frame_idx")["timestamp_ms"].first()
    wide.insert(0, "timestamp_ms", times.reindex(wide.index).to_numpy())
    return wide.reset_index().sort_values("frame_idx").reset_index(drop=True)


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle ABC in degrees for arrays of points shaped (n, 2). NaN in -> NaN out."""
    ba, bc = a - b, c - b
    dot = np.einsum("ij,ij->i", ba, bc)
    norm = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.clip(dot / norm, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _xy(df: pd.DataFrame, joint: str) -> np.ndarray:
    cols = [f"{joint}_x", f"{joint}_y"]
    if not set(cols) <= set(df.columns):
        return np.full((len(df), 2), np.nan)
    return df[cols].to_numpy(dtype=float)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end_exclusive) index pairs of consecutive True values."""
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    return list(zip(edges[::2], edges[1::2]))


def _speed(xy: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    """Speed per frame (units per second); NaN where position is missing."""
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    dt = np.diff(t_s)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = d / dt
    return np.concatenate([[np.nan], v])


# ---------------------------------------------------------------- frame metrics

def frame_metrics(keypoints: pd.DataFrame, cfg: MetricConfig | None = None) -> pd.DataFrame:
    """Per-frame metrics for one clip (one model's keypoints)."""
    cfg = cfg or MetricConfig()
    wide = to_wide(keypoints, cfg.min_confidence)
    if wide.empty:
        return wide

    t_s = wide["timestamp_ms"].to_numpy(dtype=float) / 1000.0
    fps = 1.0 / np.median(np.diff(t_s)) if len(t_s) > 1 else 30.0
    gap = max(1, int(round(cfg.max_gap_s * fps)))
    win = max(1, int(round(cfg.smooth_window_s * fps)))

    coords = [c for c in wide.columns if c.endswith(("_x", "_y"))]
    wide[coords] = (wide[coords]
                    .interpolate(limit=gap, limit_area="inside")
                    .rolling(win, center=True, min_periods=1).mean())

    sh_mid = (_xy(wide, "left_shoulder") + _xy(wide, "right_shoulder")) / 2
    hip_mid = (_xy(wide, "left_hip") + _xy(wide, "right_hip")) / 2
    torso = np.linalg.norm(sh_mid - hip_mid, axis=1)
    scale = float(np.nanmedian(torso)) if np.isfinite(torso).any() else np.nan

    out = pd.DataFrame({
        "frame_idx": wide["frame_idx"],
        "timestamp_ms": wide["timestamp_ms"],
        "t_s": t_s - t_s[0],
        "body_scale_px": scale,
        "torso_px": torso,
    })
    for side in ("left", "right"):
        out[f"{side}_elbow_deg"] = joint_angle(
            _xy(wide, f"{side}_shoulder"), _xy(wide, f"{side}_elbow"), _xy(wide, f"{side}_wrist"))
        out[f"{side}_knee_deg"] = joint_angle(
            _xy(wide, f"{side}_hip"), _xy(wide, f"{side}_knee"), _xy(wide, f"{side}_ankle"))
        out[f"{side}_arm_straight"] = (out[f"{side}_elbow_deg"] >= cfg.straight_arm_deg).where(
            out[f"{side}_elbow_deg"].notna())

        foot_name = next((n for n in FOOT_POINT[side] if f"{n}_x" in wide.columns), None)
        foot = _xy(wide, foot_name) / scale if foot_name else np.full((len(wide), 2), np.nan)
        out[f"{side}_foot_x_tl"], out[f"{side}_foot_y_tl"] = foot[:, 0], foot[:, 1]
        out[f"{side}_foot_speed_tlps"] = _speed(foot, t_s)

    hip = hip_mid / scale
    out["hip_x_tl"], out["hip_y_tl"] = hip[:, 0], hip[:, 1]
    out["hip_speed_tlps"] = _speed(hip, t_s)
    return out


# ---------------------------------------------------------------- attempt metrics

def _foot_moves(frames: pd.DataFrame, side: str, cfg: MetricConfig) -> tuple[int, int]:
    """(moves, readjustments) for one foot.

    A move = a run of frames faster than foot_move_speed. Moves separated by less
    than foot_still_s of stillness are merged. A move whose start-to-end distance
    is under readjust_max_dist torso lengths counts as a readjustment.
    """
    speed = frames[f"{side}_foot_speed_tlps"].to_numpy()
    if np.isnan(speed).all():
        return 0, 0
    t = frames["t_s"].to_numpy()
    runs = _runs(np.nan_to_num(speed) > cfg.foot_move_speed)
    merged: list[list[int]] = []
    for start, end in runs:
        if merged and t[start] - t[merged[-1][1] - 1] < cfg.foot_still_s:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    pos = frames[[f"{side}_foot_x_tl", f"{side}_foot_y_tl"]].to_numpy()
    readjust = 0
    for start, end in merged:
        a, b = pos[max(start - 1, 0)], pos[min(end, len(pos) - 1)]
        if np.isfinite(a).all() and np.isfinite(b).all() and \
                np.linalg.norm(b - a) < cfg.readjust_max_dist:
            readjust += 1
    return len(merged), readjust


def _pauses(frames: pd.DataFrame, cfg: MetricConfig) -> list[float]:
    """Durations (s) of stretches where the hips stay slower than pause_speed."""
    speed = frames["hip_speed_tlps"].to_numpy()
    t = frames["t_s"].to_numpy()
    out = []
    for start, end in _runs(np.nan_to_num(speed, nan=np.inf) < cfg.pause_speed):
        dur = t[end - 1] - t[start]
        if dur >= cfg.pause_min_s:
            out.append(float(dur))
    return out


def _hip_path(frames: pd.DataFrame) -> tuple[float, float, float]:
    """(path length TL, efficiency 0-1, mean |jerk| TL/s^3) of the hip trajectory."""
    hip = frames[["hip_x_tl", "hip_y_tl"]].dropna()
    if len(hip) < 4:
        return np.nan, np.nan, np.nan
    t = frames.loc[hip.index, "t_s"].to_numpy()
    xy = hip.to_numpy()
    length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    straight = float(np.linalg.norm(xy[-1] - xy[0]))
    efficiency = straight / length if length > 0 else np.nan
    vel = np.gradient(xy, t, axis=0)
    acc = np.gradient(vel, t, axis=0)
    jerk = np.gradient(acc, t, axis=0)
    return length, efficiency, float(np.nanmean(np.linalg.norm(jerk, axis=1)))


def attempt_metrics(frames: pd.DataFrame, cfg: MetricConfig | None = None) -> dict:
    """One summary row for a clip, from its frame metrics."""
    cfg = cfg or MetricConfig()
    if frames.empty:
        return {"frames": 0}
    row: dict = {
        "frames": len(frames),
        "duration_s": float(frames["t_s"].iloc[-1]),
        "body_scale_px": float(frames["body_scale_px"].iloc[0]),
    }
    for side in ("left", "right"):
        straight = frames[f"{side}_arm_straight"].dropna()
        row[f"{side}_arm_straight_share"] = float(straight.mean()) if len(straight) else np.nan
        row[f"{side}_elbow_median_deg"] = float(frames[f"{side}_elbow_deg"].median())
        moves, readjust = _foot_moves(frames, side, cfg)
        row[f"{side}_foot_moves"], row[f"{side}_foot_readjustments"] = moves, readjust

    both = frames[["left_arm_straight", "right_arm_straight"]].dropna()
    row["arms_straight_share"] = float(both.stack().astype(float).mean()) if len(both) else np.nan
    row["foot_moves"] = row["left_foot_moves"] + row["right_foot_moves"]
    row["foot_readjustments"] = row["left_foot_readjustments"] + row["right_foot_readjustments"]

    pauses = _pauses(frames, cfg)
    row["pauses"] = len(pauses)
    row["pause_total_s"] = float(sum(pauses))
    row["pause_longest_s"] = float(max(pauses)) if pauses else 0.0

    length, efficiency, jerk = _hip_path(frames)
    row["hip_path_tl"], row["hip_path_efficiency"], row["hip_mean_jerk"] = length, efficiency, jerk
    return row
