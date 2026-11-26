"""Your verdicts on the feedback flags: is each red moment a real mistake?

The review log is metadata/flag_reviews.csv, one row per flag (or per moment the
system missed):

| Column | Meaning |
| --- | --- |
| video_id | the clip |
| flag | left_arm_bent, right_arm_bent, left_foot_readjust, right_foot_readjust, pause, or other |
| start, end | clip time as m:ss.s (as QuickTime shows it) or seconds |
| verdict | was the detection right? correct (it happened), wrong (it didn't), unsure, missed (you add these) |
| mistake | was it a mistake? yes, no (needed or harmless: lock-off, foot swap, no load), blank = not judged |
| note | free text: why, what you'd coach |
| source | auto = exported from a feedback run; you = a row you added |

`export` adds every flag from the latest feedback runs with a blank verdict and keeps
everything you already wrote. `score` compares your verdicts with the current flags,
so after changing a threshold you can see how many wrong flags went away and how many
correct ones were lost.

Rows are matched by clip, flag type and overlapping time, so a flag that moved a few
frames after retuning keeps its verdict.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from boulder.ingest.attempts import parse_seconds

REVIEW_PATH = Path("metadata/flag_reviews.csv")
FEEDBACK_ROOT = Path("data/processed/feedback")
COLUMNS = ["video_id", "flag", "start", "end", "verdict", "mistake", "note", "source"]
MISTAKE_VALUES = {"", "yes", "no"}
VERDICTS = {"correct", "wrong", "unsure", "missed"}
FLAG_TYPES = {"left_arm_bent", "right_arm_bent", "left_foot_readjust", "right_foot_readjust",
              "pause", "other"}


def fmt_time(seconds: float) -> str:
    """64.25 -> '1:04.3' (the m:ss.s form QuickTime shows)."""
    tenths = int(round(seconds * 10))
    return f"{tenths // 600}:{(tenths % 600) / 10:04.1f}"


def read_reviews(path: Path = REVIEW_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return df.reindex(columns=COLUMNS, fill_value="")


def write_reviews(df: pd.DataFrame, path: Path = REVIEW_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df[COLUMNS].to_csv(path, index=False)


def load_events(root: Path = FEEDBACK_ROOT, model: str = "rtmpose-feet") -> pd.DataFrame:
    """Every flag from the latest feedback runs: video_id, flag, start_s, end_s."""
    files = sorted(root.glob(f"camera=*/model={model}/*.parquet"))
    if not files:
        return pd.DataFrame(columns=["video_id", "flag", "start_s", "end_s"])
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def clean_reviews(reviews: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Parse times and check values. Adds start_s / end_s; returns the rows and a list of issues."""
    extra = [c for c in reviews.columns if c not in COLUMNS]
    out = reviews.reindex(columns=COLUMNS + extra, fill_value="")  # older files lack 'mistake'
    out[COLUMNS] = out[COLUMNS].fillna("").astype(str)
    issues = []
    starts, ends = [], []
    for i, r in enumerate(out.itertuples(index=False)):
        where = f"flag_reviews.csv line {i + 2}"
        s = parse_seconds(r.start) if r.start else None
        e = parse_seconds(r.end) if r.end else None
        if s is None:
            issues.append(f"{where}: start must be m:ss.s or seconds, got '{r.start}'")
        if e is None:
            e = s  # a missed moment can be a single time
        if s is not None and e is not None and e < s:
            issues.append(f"{where}: end is before start")
        if r.verdict and r.verdict.lower() not in VERDICTS:
            issues.append(f"{where}: verdict must be one of {sorted(VERDICTS)}, got '{r.verdict}'")
        if r.mistake and r.mistake.lower() not in MISTAKE_VALUES:
            issues.append(f"{where}: mistake must be yes, no or blank, got '{r.mistake}'")
        if r.flag not in FLAG_TYPES:
            issues.append(f"{where}: flag must be one of {sorted(FLAG_TYPES)}, got '{r.flag}'")
        starts.append(s if s is not None else math.nan)
        ends.append(e if e is not None else math.nan)
    out["start_s"], out["end_s"] = starts, ends
    out["verdict"] = out["verdict"].str.lower().str.strip()
    return out, issues


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float, pad: float = 0.2) -> bool:
    return a_start <= b_end + pad and b_start <= a_end + pad


def match(reviews: pd.DataFrame, events: pd.DataFrame) -> list[int | None]:
    """For each review row, the index of an overlapping current event (same clip and flag)."""
    out: list[int | None] = []
    for r in reviews.itertuples(index=False):
        hits = events[(events["video_id"] == r.video_id) & (events["flag"] == r.flag)]
        found = None
        for idx, e in hits.iterrows():
            if not math.isnan(r.start_s) and _overlaps(r.start_s, r.end_s, e.start_s, e.end_s):
                found = idx
                break
        out.append(found)
    return out


def merge_events(reviews: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Add current events that no review row covers; drop auto rows nobody reviewed that no
    longer exist. Your verdicts, notes and added rows are always kept."""
    reviews, _ = clean_reviews(reviews)
    matched = match(reviews, events)
    covered = {m for m in matched if m is not None}
    stale = [(row.source == "auto" and not row.verdict and not row.note and not row.mistake
              and m is None)
             for row, m in zip(reviews.itertuples(index=False), matched)]
    kept = reviews.loc[[not s for s in stale]] if len(reviews) else reviews

    new = events.drop(index=list(covered))
    new_rows = pd.DataFrame({
        "video_id": new["video_id"], "flag": new["flag"],
        "start": new["start_s"].map(fmt_time), "end": new["end_s"].map(fmt_time),
        "verdict": "", "mistake": "", "note": "", "source": "auto",
    })
    merged = pd.concat([kept[COLUMNS], new_rows], ignore_index=True)
    merged["_t"] = [parse_seconds(t) or 0.0 for t in merged["start"]]
    merged = merged.sort_values(["video_id", "_t"], kind="stable").drop(columns="_t")
    return merged.reset_index(drop=True), {"added": len(new_rows), "dropped": sum(stale),
                                            "kept": len(kept)}


def score(reviews: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Per flag type: how your verdicts line up with the current flags.

    correct, wrong      your verdicts on flags the current settings still show
    precision           correct / (correct + wrong): share of shown flags that are real mistakes
    correct_lost        flags you called correct that the current settings no longer show
    missed_now_caught   moments you marked missed that a current flag now covers
    """
    reviews, _ = clean_reviews(reviews)
    reviews = reviews[reviews["start_s"].notna()]
    judged = reviews[reviews["verdict"].isin(["correct", "wrong", "unsure"])]
    missed = reviews[reviews["verdict"] == "missed"]
    j_match = match(judged, events)

    # a missed moment counts as caught if any current flag of that type (or any type, for
    # 'other') overlaps it
    def caught(r) -> bool:
        hits = events[events["video_id"] == r.video_id]
        if r.flag != "other":
            hits = hits[hits["flag"] == r.flag]
        return any(_overlaps(r.start_s, r.end_s, e.start_s, e.end_s) for e in hits.itertuples())

    rows = []
    for flag in sorted(FLAG_TYPES):
        ev = events[events["flag"] == flag]
        sel = judged["flag"] == flag
        verdicts = judged.loc[sel, "verdict"].tolist()
        shown = [m is not None for m, s in zip(j_match, sel) if s]
        correct_now = sum(v == "correct" and s for v, s in zip(verdicts, shown))
        wrong_now = sum(v == "wrong" and s for v, s in zip(verdicts, shown))
        m = missed[missed["flag"] == flag]
        row = {
            "flag": flag,
            "flags_now": len(ev),
            "reviewed_now": sum(shown),
            "correct": correct_now,
            "wrong": wrong_now,
            "precision": correct_now / (correct_now + wrong_now)
            if correct_now + wrong_now else math.nan,
            "correct_lost": sum(v == "correct" and not s for v, s in zip(verdicts, shown)),
            "missed": len(m),
            "missed_now_caught": sum(caught(r) for r in m.itertuples(index=False)),
        }
        if row["flags_now"] or len(verdicts) or len(m):
            rows.append(row)
    return pd.DataFrame(rows)
