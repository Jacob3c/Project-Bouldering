"""Tests for clip file-name parsing and date normalization."""

from boulder.ingest.clip_name import normalize_date, parse_clip_name


def test_normalize_date_pads_month_and_day():
    assert normalize_date("2026-1-9") == "2026-01-09"


def test_normalize_date_keeps_padded_date():
    assert normalize_date("2026-01-19") == "2026-01-19"


def test_normalize_date_rejects_impossible_date():
    assert normalize_date("2026-13-45") is None
    assert normalize_date("2026-02-30") is None


def test_unpadded_and_padded_names_share_canonical_id():
    short = parse_clip_name("2026-1-19_ABP_Orange-V3_att01_send_handheld.MOV")
    padded = parse_clip_name("2026-01-19_ABP_Orange-V3_att01_send_handheld.MOV")
    assert short.canonical_id == padded.canonical_id
    assert short.canonical_id == "2026-01-19_ABP_Orange-V3_att01_send_handheld"
    assert short.date == "2026-01-19"
    # video_id is unchanged: it still matches the real file name.
    assert short.video_id == "2026-1-19_ABP_Orange-V3_att01_send_handheld"


def test_impossible_date_marks_name_bad():
    info = parse_clip_name("2026-13-45_ABP_Orange-V3_att01_send_handheld.MOV")
    assert info.name_ok is False
    assert info.canonical_id is None


# --- gym, colour, grade and wall-style normalization ---

from boulder.ingest.aliases import canonical  # noqa: E402
from boulder.ingest.clip_name import normalize_grade  # noqa: E402


def test_gym_spellings_map_to_one_name():
    for raw in ["MainRec", "mainrec", "Main-Rec", "MAIN_REC"]:
        assert canonical("gyms", raw) == ("MainRec", True)


def test_colour_variant_maps_to_official():
    assert canonical("colours", "grey") == ("Gray", True)
    assert canonical("colours", "ORANGE") == ("Orange", True)


def test_wall_style_variant_maps_to_official():
    assert canonical("wall_styles", "Steep") == ("overhang", True)


def test_unknown_name_is_kept_and_flagged():
    assert canonical("gyms", " NewGym ") == ("NewGym", False)


def test_normalize_grade():
    assert normalize_grade("v3") == "V3"
    assert normalize_grade("v03-4") == "V3-4"
    assert normalize_grade("vb") == "VB"
    assert normalize_grade("V5-3") is None
    assert normalize_grade("5a") is None


def test_messy_name_gets_clean_canonical_id():
    info = parse_clip_name("2026-1-19_abp_orange-v3_att1_SEND_Handheld.MOV")
    assert info.canonical_id == "2026-01-19_ABP_Orange-V3_att01_send_handheld"
    assert info.issues == ""


def test_unknown_gym_listed_in_issues():
    info = parse_clip_name("2026-01-19_NewGym_Orange-V3_att01_send_handheld.MOV")
    assert info.name_ok is True
    assert "unknown gym 'NewGym'" in info.issues
