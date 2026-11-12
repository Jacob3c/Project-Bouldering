"""Tests for joining the hand-written attempts log onto the manifest."""

import pandas as pd
import pytest

from boulder.ingest.attempts import LOG_COLUMNS, build_clips, clean_log, read_log

SEND = "2026-01-19_ABP_Orange-V3_att01_send_handheld"
FELL = "2026-11-08_SouthSide_Orange-V5-6_att02_fell_tripod"


def _log(*rows: dict) -> pd.DataFrame:
    """Build a log table; any column not given is blank, like an empty CSV cell."""
    return pd.DataFrame([{c: r.get(c, "") for c in LOG_COLUMNS} for r in rows])


def _manifest(*clip_ids: str) -> pd.DataFrame:
    return pd.DataFrame({"clip_id": list(clip_ids), "fps": [30.0] * len(clip_ids)})


def test_styles_and_sections_are_normalized():
    clean, issues = clean_log(_log({"file_name": f"{SEND}.MOV", "wall_style": "Steep",
                                    "wall_sections": "vert > steep>SLAB"}))
    assert issues == []
    row = clean.iloc[0]
    assert row["wall_style"] == "overhang"
    assert row["wall_sections"] == "vertical>overhang>slab"


def test_messy_file_name_matches_canonical_id():
    clean, _ = clean_log(_log({"file_name": "2026-1-19_abp_orange-v3_att1_send_handheld.mov"}))
    assert clean["clip_id"].tolist() == [SEND]


def test_blank_cells_become_missing_values():
    clean, issues = clean_log(_log({"file_name": f"{SEND}.MOV"}))
    assert issues == []
    row = clean.iloc[0]
    assert row["wall_style"] is None and row["notes"] is None
    assert pd.isna(row["fall_move"])


def test_unknown_style_is_kept_and_reported():
    clean, issues = clean_log(_log({"file_name": f"{SEND}.MOV", "wall_style": "cave-ish"}))
    assert clean.iloc[0]["wall_style"] == "cave-ish"
    assert any("unknown wall style 'cave-ish'" in i for i in issues)


def test_fall_move_rules():
    clean, issues = clean_log(_log({"file_name": f"{FELL}.mp4", "fall_move": "4"}))
    assert issues == [] and clean.iloc[0]["fall_move"] == 4

    _, issues = clean_log(_log({"file_name": f"{FELL}.mp4", "fall_move": "four"}))
    assert any("whole number" in i for i in issues)

    _, issues = clean_log(_log({"file_name": f"{SEND}.MOV", "fall_move": "3"}))
    assert any("is a send" in i for i in issues)


def test_bad_file_name_and_duplicates_are_reported():
    clean, issues = clean_log(_log({"file_name": "random.mp4"},
                                   {"file_name": f"{SEND}.MOV", "notes": "first"},
                                   {"file_name": f"{SEND}.MOV", "notes": "second"}))
    assert any("line 2: cannot read file name" in i for i in issues)
    assert any("more than once" in i for i in issues)
    assert clean["notes"].tolist() == ["first"]


def test_left_join_keeps_every_clip_and_flags_unmatched_rows():
    other = "2026-09-12_Stone_CO_Purple-V3_att01_send_handheld"
    clips, issues = build_clips(
        _manifest(SEND, other),
        _log({"file_name": f"{SEND}.MOV", "wall_style": "slab"},
             {"file_name": f"{FELL}.mp4", "wall_style": "roof"}),
    )
    assert clips["clip_id"].tolist() == [SEND, other]          # every ingested clip, in order
    assert clips["wall_style"].tolist()[0] == "slab"
    assert pd.isna(clips["wall_style"].tolist()[1])            # no log row -> blank
    assert any(f"{FELL} is in the log but was never ingested" in i for i in issues)


def test_read_log_requires_all_columns(tmp_path):
    path = tmp_path / "attempts.csv"
    path.write_text("file_name,wall_style\nx.mp4,slab\n")
    with pytest.raises(ValueError, match="missing columns"):
        read_log(path)


def test_read_log_keeps_blank_cells_as_text(tmp_path):
    path = tmp_path / "attempts.csv"
    path.write_text("file_name,wall_style,wall_sections,fall_move,notes\n"
                    f" {SEND}.MOV ,,,, \n")
    log = read_log(path)
    assert log.iloc[0].tolist() == [f"{SEND}.MOV", "", "", "", ""]


def test_real_log_file_parses_cleanly():
    """metadata/attempts.csv in the repo must always be readable and issue-free."""
    _, issues = clean_log(read_log())
    assert issues == []
