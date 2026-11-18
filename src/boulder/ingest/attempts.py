"""Join the hand-written attempts log onto the manifest: one row per clip.

Some facts can't come from a file name or a video: the wall style, the
sections of wall the climb passes through, the move a fall happened on, and
free-form notes. You type those into metadata/attempts.csv; this step cleans
them and joins them onto data/manifest.parquet (written by boulder.ingest.run).

Usage:
    python -m boulder.ingest.attempts            # write data/clips.parquet
    python -m boulder.ingest.attempts --strict   # exit with an error if any issue is found

Log columns (all optional except file_name; blank cells are fine):
    file_name      the clip's file name; any spelling the name parser accepts
    climb_start_s  when the climb starts (leaving the ground), in seconds (4.2) or
                   m:ss as QuickTime shows it (0:04.2)
    climb_end_s    when the climb ends (top-out or fall), same formats
    wall_style     main style of the climb: slab | vertical | overhang | roof | arete | dihedral
    wall_sections  styles in climbing order, joined with ">", e.g. vertical>overhang>slab
    fall_move      move number of the fall (1 = first move; start or 0 = fell on the
                   start holds); blank on a send
    notes          free text, never used in calculations

Styles are mapped to official names via configs/aliases.yaml ("steep" -> "overhang").
The join is a LEFT join: every ingested clip appears, with empty log columns if
it has no row in the log.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

from boulder.ingest.aliases import canonical
from boulder.ingest.clip_name import parse_clip_name
from boulder.ingest.run import MANIFEST_PATH, load_manifest, save_manifest

LOG_PATH = Path("metadata/attempts.csv")
CLIPS_PATH = Path("data/clips.parquet")
LOG_COLUMNS = ["file_name", "climb_start_s", "climb_end_s", "wall_style", "wall_sections",
               "fall_move", "notes"]
CLEAN_COLUMNS = ["clip_id", "climb_start_s", "climb_end_s", "wall_style", "wall_sections",
                 "fall_move", "notes"]
DURATION_SLACK_S = 0.5  # climb times may run this far past the reported clip length
START_MOVE = 0  # fall_move for a fall on the start holds, before move 1


def read_log(path: Path = LOG_PATH) -> pd.DataFrame:
    """Read the log as plain text: every cell a string, blank cells as ''."""
    log = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in LOG_COLUMNS if c not in log.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    return log[LOG_COLUMNS].apply(lambda col: col.str.strip())


def _style(raw: str, where: str, issues: list[str]) -> str:
    """Official style name; unknown names are kept as typed and reported."""
    official, known = canonical("wall_styles", raw)
    if not known:
        issues.append(f"{where}: unknown wall style '{raw}' (add it to configs/aliases.yaml?)")
    return official


def parse_fall_move(raw: str) -> int | None:
    """Fall move as a number: 'start' or '0' -> 0, '4' -> 4, anything else -> None.

    Stored as a number (not the word "start") so the column stays numeric and
    falls can be sorted or compared by how far the climb got.
    """
    text = raw.strip().lower()
    if text == "start":
        return START_MOVE
    return int(text) if text.isdigit() else None


def parse_seconds(raw: str) -> float | None:
    """A time in seconds: '4.2' -> 4.2, '0:38' -> 38.0, '1:05.5' -> 65.5; else None."""
    text = raw.strip()
    minutes, _, seconds = text.rpartition(":")
    try:
        total = float(seconds) + (int(minutes) * 60 if minutes else 0)
    except ValueError:
        return None
    if not math.isfinite(total) or total < 0 or (minutes and not 0 <= float(seconds) < 60):
        return None
    return total


def _climb_times(rec: dict, where: str, issues: list[str]) -> tuple[float | None, float | None]:
    """Parse and sanity-check the climb window of one log row."""
    times = []
    for col in ("climb_start_s", "climb_end_s"):
        value = parse_seconds(rec[col]) if rec[col] else None
        if rec[col] and value is None:
            issues.append(f"{where}: {col} must be seconds (4.2) or m:ss (0:04.2), "
                          f"got '{rec[col]}'")
        times.append(value)
    start, end = times
    if start is not None and end is not None and start >= end:
        issues.append(f"{where}: climb_start_s ({start}) must be before climb_end_s ({end})")
    return start, end


def clean_log(log: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Turn raw log rows into one clean row per clip_id, plus a list of problems found."""
    rows, issues = [], []
    for i, rec in enumerate(log.to_dict("records")):
        where = f"attempts.csv line {i + 2}"  # +2: header is line 1, rows start at 2
        info = parse_clip_name(rec["file_name"])
        if info.canonical_id is None:
            issues.append(f"{where}: cannot read file name '{rec['file_name']}'")
            continue

        style = _style(rec["wall_style"], where, issues) if rec["wall_style"] else None
        sections = None
        if rec["wall_sections"]:
            parts = [p.strip() for p in rec["wall_sections"].split(">") if p.strip()]
            sections = ">".join(_style(p, where, issues) for p in parts)

        climb_start, climb_end = _climb_times(rec, where, issues)
        fall_move = parse_fall_move(rec["fall_move"]) if rec["fall_move"] else None
        if rec["fall_move"] and fall_move is None:
            issues.append(f"{where}: fall_move must be 'start' or a whole number, got "
                          f"'{rec['fall_move']}'")
        if fall_move is not None and info.outcome == "send":
            issues.append(f"{where}: fall_move given but the clip is a send")

        rows.append({"clip_id": info.canonical_id, "climb_start_s": climb_start,
                     "climb_end_s": climb_end, "wall_style": style,
                     "wall_sections": sections, "fall_move": fall_move,
                     "notes": rec["notes"] or None})

    clean = pd.DataFrame(rows, columns=CLEAN_COLUMNS)
    dupes = clean["clip_id"][clean["clip_id"].duplicated()].unique()
    for clip_id in dupes:
        issues.append(f"{clip_id} appears more than once in the log; keeping the first row")
    clean = clean.drop_duplicates("clip_id", keep="first")
    clean["fall_move"] = clean["fall_move"].astype("Int64")  # whole numbers that can be blank
    for col in ("climb_start_s", "climb_end_s"):
        clean[col] = clean[col].astype("float64")             # blank -> NaN
    return clean.reset_index(drop=True), issues


def build_clips(manifest: pd.DataFrame, log: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Left-join the cleaned log onto the manifest: every ingested clip, one row each."""
    clean, issues = clean_log(log)
    unmatched = sorted(set(clean["clip_id"]) - set(manifest["clip_id"]))
    for clip_id in unmatched:
        issues.append(f"{clip_id} is in the log but was never ingested")
    clips = manifest.merge(clean, on="clip_id", how="left", validate="one_to_one")
    if "duration_s" in clips.columns:
        for col in ("climb_start_s", "climb_end_s"):
            too_late = clips[col] > clips["duration_s"] + DURATION_SLACK_S
            for _, row in clips[too_late].iterrows():
                issues.append(f"{row['clip_id']}: {col} {row[col]:.1f} s is after the end of "
                              f"the clip ({row['duration_s']:.1f} s)")
    return clips, issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, default=LOG_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--out", type=Path, default=CLIPS_PATH)
    parser.add_argument("--strict", action="store_true", help="exit 1 if any issue is found")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    if manifest.empty:
        sys.exit(f"no manifest at {args.manifest}: run python -m boulder.ingest.run first")
    clips, issues = build_clips(manifest, read_log(args.log))
    save_manifest(clips, args.out)

    filled = clips["wall_style"].notna().sum()
    print(f"wrote {args.out}: {len(clips)} clips, {filled} with a wall style")
    for issue in issues:
        print("issue:", issue)
    if issues and args.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
