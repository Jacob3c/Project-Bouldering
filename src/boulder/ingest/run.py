"""Ingest raw climbing videos: validate the name, copy under a canonical ID, log a manifest.

Ingest is the pipeline's entry gate. Every later stage reads clips from
data/raw/, so names are cleaned once here and nowhere else.

Usage:
    python -m boulder.ingest.run data/test/*.MOV           # ingest these clips
    python -m boulder.ingest.run data/test                 # every video in a folder
    python -m boulder.ingest.run data/test --dry-run       # report only, copy nothing

For each clip:
    1. Parse the file name (boulder.ingest.clip_name). Names that don't match the
       pattern, or that have issues (unknown gym/colour, bad grade), are rejected.
    2. Hash the file bytes (SHA-256). A clip whose hash is already in the
       manifest is a duplicate, even under a different name, and is skipped.
    3. Copy it to data/raw/<canonical_id><ext>. The original is left in place.
    4. Read fps, frame count and size, and add one row to data/manifest.parquet.

Running it twice does nothing the second time (idempotent).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd

from boulder.ingest.clip_name import parse_clip_name

RAW_DIR = Path("data/raw")
MANIFEST_PATH = Path("data/manifest.parquet")
VIDEO_EXTS = {".mov", ".mp4", ".avi", ".m4v"}
NAME_PATTERN = "<date>_<gym>_<colour>-<grade>_att<NN>_<outcome>_<camera>"

# Possible outcomes for one clip
INGESTED = "ingested"
DUPLICATE = "duplicate"      # same bytes already ingested
ALREADY = "already"          # same canonical ID and same bytes: nothing to do
CONFLICT = "id_conflict"     # same canonical ID, different bytes
REJECTED = "rejected"        # bad file name


@dataclass
class Result:
    source: Path
    status: str
    clip_id: str | None = None
    detail: str = ""


def file_sha256(path: Path, chunk_bytes: int = 1 << 20) -> str:
    """SHA-256 of a file, read 1 MB at a time so big videos never fill memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def probe_video(path: Path) -> dict:
    """fps, frame count, size and duration as OpenCV reports them."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"could not open video {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or None
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        return {
            "fps": fps,
            "n_frames": n_frames,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration_s": n_frames / fps if fps and n_frames else None,
        }
    finally:
        cap.release()


def find_videos(inputs: list[Path]) -> list[Path]:
    """Expand folders into the videos inside them; keep files as given."""
    found: list[Path] = []
    for item in inputs:
        if item.is_dir():
            found += sorted(p for p in item.iterdir() if p.suffix.lower() in VIDEO_EXTS)
        elif item.suffix.lower() in VIDEO_EXTS:
            found.append(item)
    return found


def load_manifest(path: Path = MANIFEST_PATH) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def save_manifest(df: pd.DataFrame, path: Path = MANIFEST_PATH) -> None:
    """Write to a temp file, then rename: a crash mid-write never leaves a broken manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def classify(clip_id: str, sha256: str, manifest: pd.DataFrame) -> tuple[str, str]:
    """Decide what to do with a valid clip, given what is already ingested."""
    if manifest.empty:
        return INGESTED, ""
    same_id = manifest[manifest["clip_id"] == clip_id]
    if not same_id.empty:
        if same_id["sha256"].iloc[0] == sha256:
            return ALREADY, ""
        return CONFLICT, "a different video already has this ID"
    same_bytes = manifest[manifest["sha256"] == sha256]
    if not same_bytes.empty:
        return DUPLICATE, f"same video as {same_bytes['clip_id'].iloc[0]}"
    return INGESTED, ""


def ingest(inputs: list[Path], raw_dir: Path = RAW_DIR, manifest_path: Path = MANIFEST_PATH,
           dry_run: bool = False) -> list[Result]:
    manifest = load_manifest(manifest_path)
    new_rows: list[dict] = []
    results: list[Result] = []

    for source in find_videos(inputs):
        info = parse_clip_name(source)
        if not info.name_ok or info.issues or info.canonical_id is None:
            reason = info.issues or f"name does not match {NAME_PATTERN}"
            results.append(Result(source, REJECTED, detail=reason))
            continue

        sha256 = file_sha256(source)
        seen = pd.concat([manifest, pd.DataFrame(new_rows)], ignore_index=True)
        status, detail = classify(info.canonical_id, sha256, seen)
        results.append(Result(source, status, info.canonical_id, detail))
        if status != INGESTED or dry_run:
            continue

        dest = raw_dir / f"{info.canonical_id}{source.suffix.lower()}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        new_rows.append({
            "clip_id": info.canonical_id,
            **{k: v for k, v in info.to_dict().items()
               if k not in ("video_id", "canonical_id", "name_ok", "issues")},
            "source_name": source.name,
            "raw_path": dest.as_posix(),
            "sha256": sha256,
            "size_bytes": source.stat().st_size,
            **probe_video(dest),
            "ingested_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        })

    if new_rows and not dry_run:
        save_manifest(pd.concat([manifest, pd.DataFrame(new_rows)], ignore_index=True),
                      manifest_path)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="video files or folders")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--dry-run", action="store_true", help="report only, copy nothing")
    args = parser.parse_args()

    results = ingest(args.inputs, args.raw_dir, args.manifest, args.dry_run)
    if not results:
        print("no videos found")
    for r in results:
        line = f"{r.status:<12} {r.source.name}"
        if r.clip_id and r.clip_id != r.source.stem:
            line += f" -> {r.clip_id}"
        if r.detail:
            line += f"  ({r.detail})"
        print(line)
    counts = pd.Series([r.status for r in results]).value_counts().to_dict() if results else {}
    print("summary:", ", ".join(f"{n} {s}" for s, n in counts.items()) or "nothing to do",
          "(dry run)" if args.dry_run else "")


if __name__ == "__main__":
    main()
