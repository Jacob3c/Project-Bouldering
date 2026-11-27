"""Tune the feedback rules against your reviewed flags.

Usage:
    python -m boulder.review.calibrate             # foot readjust, then bent arm
    python -m boulder.review.calibrate --top 20

Settings are tuned on "tune" clips only; the best one is then scored on the "test"
clips it never saw (boulder.review.splits). A big drop on test = overfit.

Your verdicts in metadata/flag_reviews.csv say whether each flag's detection was right
("correct") or not ("wrong"). This tool:

1. features   for every reviewed foot flag, measures the motion behind it (how far the
              foot moved, how fast, whether the other foot or the whole body moved too)
              and shows which measures separate correct from wrong flags. AUC 0.5 = no
              help, 1.0 (or 0.0) = separates them perfectly.
2. grid       re-runs the foot rule with many settings on the saved per-frame metrics and
              counts how many of your correct flags each setting keeps and how many wrong
              ones it still shows. Flags nobody reviewed (new with a setting) are counted
              as "unreviewed": check a few of those before trusting a setting.

Reads data/processed/metrics/frames/... (boulder.transforms.run) and the attempts rows
for the MetricConfig used. Changes nothing; copy the setting you like into MetricConfig.
"""

from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from boulder.render.feedback import FeedbackConfig, feedback_flags, flag_events
from boulder.transforms.metrics import MetricConfig, foot_move_spans

from .flags import REVIEW_PATH, clean_reviews, read_reviews
from .splits import load_overrides, split_of

METRIC_ROOT = Path("data/processed/metrics")
MODEL = "rtmpose-feet"
FOOT_FLAGS = ["left_foot_readjust", "right_foot_readjust"]
ARM_FLAGS = ["left_arm_bent", "right_arm_bent"]

ARM_GRID = {
    "bent_arm_deg": [110.0, 120.0, 130.0],
    "bent_arm_min_s": [1.5, 2.0, 3.0],
    "hand_still_speed": [None, 0.3, 0.5, 0.8],
}

GRID = {
    "foot_move_speed": [0.6, 0.8, 1.0, 1.2],
    "foot_still_s": [0.3, 0.5],
    "readjust_max_dist": [0.3, 0.4],
    "foot_dominance": [1.0, 1.25, 1.5, 2.0, 3.0],
    "move_min_dist": [0.0],
}


# ---------------------------------------------------------------- features

def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Chance that a random correct flag scores higher than a random wrong one."""
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if not len(pos) or not len(neg):
        return math.nan
    greater = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(greater / (len(pos) * len(neg)))


def flag_features(frames: pd.DataFrame, flag: str, start_s: float, end_s: float) -> dict:
    """Motion measures for one foot flag, from the clip's per-frame metrics."""
    side = flag.split("_")[0]
    other = "right" if side == "left" else "left"
    t = frames["timestamp_ms"].to_numpy() / 1000
    win = frames[(t >= start_s - 0.1) & (t <= end_s + 0.1)]
    if len(win) < 2:
        return {}
    gap = win[["foot_gap_x_tl", "foot_gap_y_tl"]].to_numpy()
    foot = win[[f"{side}_foot_x_tl", f"{side}_foot_y_tl"]].to_numpy()
    valid_gap = gap[np.isfinite(gap).all(axis=1)]
    mine = win[f"{side}_foot_hiprel_speed_tlps"].to_numpy()
    theirs = win[f"{other}_foot_hiprel_speed_tlps"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.nanmedian(mine / np.maximum(theirs, 1e-3))
    return {
        "gap_moved_tl": float(np.linalg.norm(valid_gap[-1] - valid_gap[0]))
        if len(valid_gap) >= 2 else math.nan,
        "gap_path_tl": float(np.nansum(np.linalg.norm(np.diff(gap, axis=0), axis=1))),
        "peak_gap_speed": float(np.nanmax(win["foot_gap_speed_tlps"])),
        "foot_frame_speed": float(np.nanmedian(win[f"{side}_foot_speed_tlps"])),
        "this_vs_other_foot": float(ratio),
        "hip_speed": float(np.nanmedian(win["hip_speed_tlps"])),
        "foot_missing_share": float(np.isnan(foot).any(axis=1).mean()),
    }


def feature_table(labels: pd.DataFrame, frames_by_clip: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for r in labels.itertuples(index=False):
        frames = frames_by_clip.get(r.video_id)
        if frames is None:
            continue
        feats = flag_features(frames, r.flag, r.start_s, r.end_s)
        if feats:
            rows.append({"video_id": r.video_id, "flag": r.flag, "verdict": r.verdict, **feats})
    return pd.DataFrame(rows)


def feature_summary(table: pd.DataFrame) -> pd.DataFrame:
    out = []
    for col in table.columns.drop(["video_id", "flag", "verdict"]):
        pos = table.loc[table["verdict"] == "correct", col].to_numpy(dtype=float)
        neg = table.loc[table["verdict"] == "wrong", col].to_numpy(dtype=float)
        out.append({"feature": col, "median_correct": np.nanmedian(pos),
                    "median_wrong": np.nanmedian(neg), "auc": auc(pos, neg)})
    df = pd.DataFrame(out)
    df["separation"] = (df["auc"] - 0.5).abs()
    return df.sort_values("separation", ascending=False).drop(columns="separation")


# ---------------------------------------------------------------- grid search

def _overlap(a0, a1, b0, b1, pad=0.2) -> bool:
    return a0 <= b1 + pad and b0 <= a1 + pad


def evaluate(cfg: MetricConfig, labels: pd.DataFrame,
             frames_by_clip: dict[str, pd.DataFrame], fcfg: FeedbackConfig | None = None,
             flag_types: list[str] = FOOT_FLAGS) -> dict:
    """Run the rules with cfg / fcfg on every clip; compare their flags with your labels."""
    kept = {"correct": 0, "wrong": 0}
    not_mistake = 0
    unreviewed = 0
    for video_id, frames in frames_by_clip.items():
        flags = feedback_flags(frames, cfg, fcfg or FeedbackConfig())
        t0 = float(frames["timestamp_ms"].iloc[0]) / 1000 if len(frames) else 0.0
        events = flag_events(flags)
        events = events[events["flag"].isin(flag_types)]
        mine = labels[labels["video_id"] == video_id]
        for e in events.itertuples(index=False):
            s, t = e.start_s + t0, e.end_s + t0
            near = np.array([_overlap(s, t, a, b) for a, b in zip(mine["start_s"], mine["end_s"])],
                            dtype=bool)
            if not ((mine["flag"] == e.flag).to_numpy() & near).any():
                unreviewed += 1
        for r in mine.itertuples(index=False):
            ev = events[events["flag"] == r.flag]
            if any(_overlap(r.start_s, r.end_s, a + t0, b + t0)
                   for a, b in zip(ev["start_s"], ev["end_s"])):
                kept[r.verdict] = kept.get(r.verdict, 0) + 1
                if r.verdict == "correct" and getattr(r, "mistake", "") == "no":
                    not_mistake += 1
    total_correct = int((labels["verdict"] == "correct").sum())
    shown = kept["correct"] + kept["wrong"]
    return {
        "correct_kept": kept["correct"],
        "correct_total": total_correct,
        "wrong_shown": kept["wrong"],
        "not_mistake_shown": not_mistake,
        "unreviewed": unreviewed,
        "precision": kept["correct"] / shown if shown else math.nan,
        "recall": kept["correct"] / total_correct if total_correct else math.nan,
    }


def grid_search(base: MetricConfig, labels: pd.DataFrame,
                frames_by_clip: dict[str, pd.DataFrame], grid: dict = GRID) -> pd.DataFrame:
    rows = []
    keys = list(grid)
    for values in itertools.product(*grid.values()):
        cfg = replace(base, **dict(zip(keys, values)))
        if cfg.move_min_dist >= cfg.readjust_max_dist:
            continue
        rows.append({**dict(zip(keys, values)), **evaluate(cfg, labels, frames_by_clip)})
    df = pd.DataFrame(rows)
    p, r = df["precision"], df["recall"]
    df["f1"] = (2 * p * r / (p + r)).fillna(0)
    is_current = np.logical_and.reduce([df[k] == getattr(base, k) for k in keys])
    df["current"] = np.where(is_current, "<- now", "")
    return df.sort_values(["f1", "precision"], ascending=False, ignore_index=True)


def arm_grid_search(labels: pd.DataFrame, frames_by_clip: dict[str, pd.DataFrame],
                    grid: dict = ARM_GRID) -> pd.DataFrame:
    """Bent-arm settings vs your labels.

    useful = shown flags that were detected right and not marked "not a mistake"
    (lock-offs and relaxed arms are marked mistake = no). Higher useful_share = the
    red arms you see are more often worth fixing.
    """
    rows = []
    keys = list(grid)
    base = FeedbackConfig()
    for values in itertools.product(*grid.values()):
        fcfg = replace(base, **dict(zip(keys, values)))
        res = evaluate(MetricConfig(), labels, frames_by_clip, fcfg, ARM_FLAGS)
        shown = res["correct_kept"] + res["wrong_shown"]
        useful = res["correct_kept"] - res["not_mistake_shown"]
        rows.append({**dict(zip(keys, values)), "shown": shown,
                     "correct_kept": res["correct_kept"], "wrong_shown": res["wrong_shown"],
                     "not_mistake_shown": res["not_mistake_shown"], "useful": useful,
                     "useful_share": useful / shown if shown else math.nan,
                     "unreviewed": res["unreviewed"]})
    df = pd.DataFrame(rows)
    is_current = np.logical_and.reduce(
        [df[k].map(lambda v, k=k: (pd.isna(v) and getattr(base, k) is None) or v == getattr(base, k))
         for k in keys])
    df["current"] = np.where(is_current, "<- now", "")
    return df.sort_values(["useful_share", "useful"], ascending=False, ignore_index=True)


def split_clips(frames_by_clip: dict[str, pd.DataFrame]) -> tuple[dict, dict]:
    overrides = load_overrides()
    tune = {k: v for k, v in frames_by_clip.items() if split_of(k, overrides) == "tune"}
    test = {k: v for k, v in frames_by_clip.items() if split_of(k, overrides) == "test"}
    return tune, test


# ---------------------------------------------------------------- command line

def _load_frames(video_ids: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for vid in video_ids:
        matches = sorted((METRIC_ROOT / "frames").glob(f"camera=*/model={MODEL}/{vid}.parquet"))
        if matches:
            out[vid] = pd.read_parquet(matches[0])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reviews", type=Path, default=REVIEW_PATH)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()
    pd.set_option("display.width", 200)
    fmt = "{:.2f}".format

    all_labels, _ = clean_reviews(read_reviews(args.reviews))
    all_labels = all_labels[all_labels["verdict"].isin(["correct", "wrong"])]
    frames_all = _load_frames(sorted(all_labels["video_id"].unique()))
    tune, test = split_clips(frames_all)
    print(f"clips: {len(tune)} tune, {len(test)} test ({', '.join(k[:22] for k in test) or 'none'})")

    labels = all_labels[all_labels["flag"].isin(FOOT_FLAGS)]
    tune_labels = labels[labels["video_id"].isin(tune)]
    test_labels = labels[labels["video_id"].isin(test)]
    print(f"\nFOOT READJUST: {(tune_labels['verdict'] == 'correct').sum()} correct, "
          f"{(tune_labels['verdict'] == 'wrong').sum()} wrong on tune clips\n")
    print("1. which motion measures separate correct from wrong flags (tune clips)")
    print(feature_summary(feature_table(tune_labels, tune)).to_string(index=False,
                                                                         float_format=fmt))
    print(f"\n2. foot rule settings on tune clips, best {args.top} (and the current one)")
    table = grid_search(MetricConfig(), tune_labels, tune)
    show = pd.concat([table.head(args.top), table[table["current"] != ""]]).drop_duplicates()
    print(show.to_string(index=False, float_format=fmt))
    best = table.iloc[0]
    cfg = replace(MetricConfig(), **{k: best[k] for k in GRID})
    cols = ["correct_kept", "correct_total", "wrong_shown", "unreviewed", "precision"]
    now_test = evaluate(MetricConfig(), test_labels, test)
    best_test = evaluate(cfg, test_labels, test)
    print("\n3. on the TEST clips (never tuned on): current vs best")
    print(pd.DataFrame([{"setting": "current", **{c: now_test[c] for c in cols}},
                        {"setting": "best", **{c: best_test[c] for c in cols}}]
                       ).to_string(index=False, float_format=fmt))

    arm = all_labels[all_labels["flag"].isin(ARM_FLAGS)]
    arm_tune = arm[arm["video_id"].isin(tune)]
    print(f"\nBENT ARM: {len(arm_tune)} reviewed flags on tune clips, "
          f"{(arm_tune['mistake'] == 'no').sum()} marked not a mistake")
    print(f"4. bent-arm settings on tune clips, best {args.top} (and the current one)")
    atable = arm_grid_search(arm_tune, tune)
    ashow = pd.concat([atable.head(args.top), atable[atable["current"] != ""]]).drop_duplicates()
    print(ashow.to_string(index=False, float_format=fmt))


if __name__ == "__main__":
    main()
