"""Ingest tests on tiny generated videos; no real footage needed."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from boulder.ingest.run import (
    ALREADY, CONFLICT, DUPLICATE, INGESTED, REJECTED,
    classify, file_sha256, find_videos, ingest, probe_video,
)

GOOD = "2026-01-19_ABP_Orange-V3_att01_send_handheld"


def _make_video(path: Path, n_frames: int = 5, shade: int = 0, w: int = 32, h: int = 24) -> Path:
    """Write a small solid-colour video; a different shade gives different bytes."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (w, h))
    for _ in range(n_frames):
        writer.write(np.full((h, w, 3), shade, dtype=np.uint8))
    writer.release()
    return path


# --- pure functions: no parquet needed ---

def test_file_sha256_same_bytes_same_hash(tmp_path):
    a = _make_video(tmp_path / "a.avi")
    b = tmp_path / "b.avi"
    b.write_bytes(a.read_bytes())
    assert file_sha256(a) == file_sha256(b)
    assert file_sha256(a) != file_sha256(_make_video(tmp_path / "c.avi", shade=200))


def test_probe_video_reads_size_and_frames(tmp_path):
    meta = probe_video(_make_video(tmp_path / "v.avi", n_frames=5))
    assert (meta["width"], meta["height"], meta["n_frames"]) == (32, 24, 5)
    assert meta["duration_s"] == pytest.approx(5 / 30)


def test_find_videos_expands_folders_and_skips_other_files(tmp_path):
    _make_video(tmp_path / f"{GOOD}.avi")
    (tmp_path / "notes.txt").write_text("hi")
    assert [p.name for p in find_videos([tmp_path])] == [f"{GOOD}.avi"]


def test_classify_outcomes():
    manifest = pd.DataFrame([{"clip_id": GOOD, "sha256": "aaa"}])
    assert classify("new_id", "bbb", pd.DataFrame()) == (INGESTED, "")
    assert classify(GOOD, "aaa", manifest)[0] == ALREADY
    assert classify(GOOD, "zzz", manifest)[0] == CONFLICT
    assert classify("other_id", "aaa", manifest)[0] == DUPLICATE
    assert classify("other_id", "bbb", manifest)[0] == INGESTED


# --- full runs: write the manifest, so they need pyarrow ---

@pytest.fixture
def dirs(tmp_path):
    pytest.importorskip("pyarrow")
    src = tmp_path / "incoming"
    src.mkdir()
    return src, tmp_path / "raw", tmp_path / "manifest.parquet"


def test_ingest_copies_under_canonical_name(dirs):
    src, raw, manifest = dirs
    _make_video(src / "2026-1-19_abp_orange-v3_att1_send_handheld.avi")
    results = ingest([src], raw, manifest)

    assert [r.status for r in results] == [INGESTED]
    assert (raw / f"{GOOD}.avi").exists()
    row = pd.read_parquet(manifest).iloc[0]
    assert row["clip_id"] == GOOD
    assert row["gym"] == "ABP" and row["grade"] == "V3"
    assert row["source_name"] == "2026-1-19_abp_orange-v3_att1_send_handheld.avi"


def test_second_run_changes_nothing(dirs):
    src, raw, manifest = dirs
    _make_video(src / f"{GOOD}.avi")
    ingest([src], raw, manifest)
    results = ingest([src], raw, manifest)
    assert [r.status for r in results] == [ALREADY]
    assert len(pd.read_parquet(manifest)) == 1


def test_same_video_two_names_is_one_clip(dirs):
    src, raw, manifest = dirs
    first = _make_video(src / f"{GOOD}.avi")
    (src / "2026-1-19_ABP_Orange-V3_att01_send_handheld.avi").write_bytes(first.read_bytes())
    statuses = sorted(r.status for r in ingest([src], raw, manifest))
    assert statuses == [ALREADY, INGESTED]  # second name maps to the same canonical ID
    assert len(pd.read_parquet(manifest)) == 1


def test_bad_names_are_rejected_and_not_copied(dirs):
    src, raw, manifest = dirs
    _make_video(src / "random_clip.avi")
    _make_video(src / "2026-01-19_NewGym_Orange-V3_att01_send_handheld.avi", shade=50)
    results = ingest([src], raw, manifest)
    assert {r.status for r in results} == {REJECTED}
    assert any("unknown gym" in r.detail for r in results)
    assert not raw.exists() or not any(raw.iterdir())
    assert not manifest.exists()


def test_dry_run_copies_nothing(dirs):
    src, raw, manifest = dirs
    _make_video(src / f"{GOOD}.avi")
    results = ingest([src], raw, manifest, dry_run=True)
    assert [r.status for r in results] == [INGESTED]
    assert not manifest.exists()
