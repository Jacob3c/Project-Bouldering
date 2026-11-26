"""Review log: merging new flags without losing verdicts, and scoring."""

import pandas as pd
import pytest

from boulder.review.flags import clean_reviews, fmt_time, merge_events, score

V = "2026-11-23_MainRec_Orange-V5-6_att01_send_handheld"


def _events(*rows):
    return pd.DataFrame(rows, columns=["video_id", "flag", "start_s", "end_s"])


def _reviews(*rows):
    return pd.DataFrame(rows, columns=["video_id", "flag", "start", "end", "verdict", "note",
                                       "source"])


def test_fmt_time():
    assert fmt_time(4.24) == "0:04.2"
    assert fmt_time(64.25) in ("1:04.2", "1:04.3")
    assert fmt_time(59.96) == "1:00.0"


def test_export_adds_new_flags_and_keeps_verdicts():
    reviews = _reviews(
        (V, "left_arm_bent", "0:10.0", "0:13.0", "wrong", "lock-off, needed", "auto"),
        (V, "left_foot_readjust", "0:20.0", "0:20.5", "", "", "auto"),        # no longer flagged
        (V, "other", "0:30.0", "", "missed", "late hip turn", "you"),
    )
    events = _events(
        (V, "left_arm_bent", 10.2, 13.4),       # same flag, moved slightly -> keeps verdict
        (V, "right_arm_bent", 15.0, 18.0),      # new
    )
    merged, counts = merge_events(reviews, events)
    assert counts == {"added": 1, "dropped": 1, "kept": 2}
    assert merged["flag"].tolist() == ["left_arm_bent", "right_arm_bent", "other"]
    assert merged.loc[0, "verdict"] == "wrong" and merged.loc[0, "note"] == "lock-off, needed"
    assert merged.loc[1, "start"] == "0:15.0" and merged.loc[1, "verdict"] == ""


def test_export_is_idempotent():
    events = _events((V, "left_arm_bent", 10.0, 13.0))
    once, _ = merge_events(_reviews(), events)
    twice, counts = merge_events(once, events)
    assert counts["added"] == 0 and len(twice) == 1


def test_clean_reviews_reports_bad_values():
    _, issues = clean_reviews(_reviews(
        (V, "left_arm_bent", "abc", "0:13", "yes", "", "auto"),
        (V, "elbow", "0:10", "0:09", "", "", "auto"),
    ))
    text = " ".join(issues)
    assert "start must be" in text and "verdict must be" in text
    assert "flag must be" in text and "end is before start" in text


def test_score_after_retune():
    reviews = _reviews(
        (V, "left_arm_bent", "0:10", "0:13", "wrong", "", "auto"),
        (V, "left_arm_bent", "0:20", "0:24", "correct", "", "auto"),
        (V, "left_arm_bent", "0:30", "0:33", "correct", "", "auto"),
        (V, "left_foot_readjust", "0:40", "0:40.5", "missed", "", "you"),
    )
    # new settings: the wrong flag is gone, one correct one lost, the missed shuffle caught
    events = _events((V, "left_arm_bent", 20.0, 24.0), (V, "left_foot_readjust", 39.9, 40.4))
    table = score(reviews, events).set_index("flag")
    arm = table.loc["left_arm_bent"]
    assert (arm["flags_now"], arm["correct"], arm["wrong"], arm["correct_lost"]) == (1, 1, 0, 1)
    assert arm["precision"] == pytest.approx(1.0)
    foot = table.loc["left_foot_readjust"]
    assert (foot["missed"], foot["missed_now_caught"]) == (1, 1)


def test_player_edits():
    from boulder.review.play import add_missed, add_note, set_verdict, todo_rows

    reviews = _reviews(
        (V, "left_arm_bent", "0:10", "0:13", "", "", "auto"),
        (V, "right_arm_bent", "0:20", "0:22", "wrong", "", "auto"),
    )
    assert todo_rows(reviews) == [0]
    assert todo_rows(reviews, include_judged=True) == [0, 1]

    reviews = set_verdict(reviews, 0, "correct")
    reviews = add_note(add_note(reviews, 0, "hung on it"), 0, "shake out instead")
    assert reviews.loc[0, "note"] == "hung on it; shake out instead"
    assert todo_rows(reviews) == []

    reviews = add_missed(reviews, V, 31.04, "lf", "foot cut")
    last = reviews.iloc[-1]
    assert (last["flag"], last["start"], last["verdict"], last["source"]) == \
        ("left_foot_readjust", "0:31.0", "missed", "you")
    assert add_missed(reviews, V, 5.0, "").iloc[-1]["flag"] == "other"
    with pytest.raises(ValueError):
        add_missed(reviews, V, 5.0, "elbow")
    assert todo_rows(reviews) == []  # rows you added are never in the review queue


def test_flag_joints_xy_picks_the_flagged_limb():
    from boulder.review.play import flag_joints_xy

    pose = {"left_elbow": (0.5, 0.5, 0.9), "left_wrist": (0.25, 0.1, 0.9),
            "right_wrist": (0.9, 0.9, 0.9), "left_big_toe": (0.1, 0.9, 0.1)}
    assert flag_joints_xy(pose, "left_arm_bent", 200, 100) == [(100, 50), (50, 10)]
    assert flag_joints_xy(pose, "left_foot_readjust", 200, 100) == []  # toe too unsure
    assert flag_joints_xy(None, "left_arm_bent", 200, 100) == []


def test_text_entry_keys():
    from boulder.review.play import TextEntry

    e = TextEntry("note")
    for ch in "no loxx":
        assert e.key(ord(ch)) is None
    e.key(127)            # backspace (Mac)
    e.key(8)              # backspace (other)
    for ch in "ad":
        e.key(ord(ch))
    assert e.text == "no load"
    assert e.key(13) == "done"
    assert TextEntry("x").key(27) == "cancel"
