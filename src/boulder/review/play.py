"""Review flags fast: each flag's moment loops in a window; one key records your verdict.

Usage:
    python -m boulder.review.play                 # every flag without a verdict
    python -m boulder.review.play --all           # also re-check flags you already judged
    python -m boulder.review.play --clip 2026-11-23_MainRec_Orange-V5-6_att01_send_handheld

Run python -m boulder.review.export first. The window plays your original clip from 1 s
before the flag to 1 s after it, over and over, with only the flag being checked drawn:
the skeleton in white, the flagged limb circled in yellow, turning red (with a red border
around the video) while the flag is on. The banner at the top names the flag, and a bar
under it shows where the flag sits in the loop.

Keys (click the video window first so it gets the key presses):
    c  correct: the detection is right     w  wrong: it didn't happen
    u  unsure                              s  skip for now
    n  add a note: type it in the video window, Enter saves, Esc cancels
    m  missed mistake at this moment: type its kind (la ra lf rf p, or Enter for other),
       then an optional note
    space  pause / play        ,  .  step one frame back / forward while paused
    r  replay from the start   b  back to the previous flag     q  save and quit

Every verdict is saved to metadata/flag_reviews.csv at once, so quitting loses nothing.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .flags import COLUMNS, FLAG_TYPES, REVIEW_PATH, clean_reviews, fmt_time, read_reviews, write_reviews

RAW_DIR = Path("data/raw")
VIEW_HEIGHT = 900  # px: clips are shrunk to this height for the review window
KEYPOINT_ROOT = Path("data/processed/keypoints")
MODEL = "rtmpose-feet"
# joints circled for each flag type
FLAG_JOINTS = {
    "left_arm_bent": ["left_elbow", "left_wrist"],
    "right_arm_bent": ["right_elbow", "right_wrist"],
    "left_foot_readjust": ["left_ankle", "left_heel", "left_big_toe"],
    "right_foot_readjust": ["right_ankle", "right_heel", "right_big_toe"],
    "pause": ["left_hip", "right_hip"],
}
TITLES = {
    "left_arm_bent": "LEFT ARM bent too long", "right_arm_bent": "RIGHT ARM bent too long",
    "left_foot_readjust": "LEFT FOOT readjust", "right_foot_readjust": "RIGHT FOOT readjust",
    "pause": "PAUSE", "other": "OTHER",
}
YELLOW, RED, WHITE, GREY = (0, 230, 255), (40, 40, 235), (255, 255, 255), (170, 170, 170)
KEY_VERDICTS = {"c": "correct", "w": "wrong", "u": "unsure"}
SHORT_FLAGS = {  # what to type for a missed mistake
    "la": "left_arm_bent", "ra": "right_arm_bent", "lf": "left_foot_readjust",
    "rf": "right_foot_readjust", "p": "pause", "": "other",
}


# ---------------------------------------------------------------- edits (tested)

def set_verdict(reviews: pd.DataFrame, i: int, verdict: str) -> pd.DataFrame:
    out = reviews.copy()
    out.loc[i, "verdict"] = verdict
    return out


def add_note(reviews: pd.DataFrame, i: int, note: str) -> pd.DataFrame:
    out = reviews.copy()
    old = out.loc[i, "note"]
    out.loc[i, "note"] = f"{old}; {note}" if old else note
    return out


def add_missed(reviews: pd.DataFrame, video_id: str, t_s: float, flag: str,
               note: str = "") -> pd.DataFrame:
    """Append a 'missed' row at clip time t_s."""
    flag = SHORT_FLAGS.get(flag, flag)
    if flag not in FLAG_TYPES:
        raise ValueError(f"unknown flag type '{flag}'")
    row = {"video_id": video_id, "flag": flag, "start": fmt_time(t_s), "end": "",
           "verdict": "missed", "mistake": "yes", "note": note, "source": "you"}
    return pd.concat([reviews, pd.DataFrame([row])[COLUMNS]], ignore_index=True)


def todo_rows(reviews: pd.DataFrame, include_judged: bool = False,
              clip: str | None = None) -> list[int]:
    """Row positions to review: flags from feedback runs, blank verdicts unless include_judged."""
    cleaned, _ = clean_reviews(reviews)
    keep = (cleaned["source"] != "you") & cleaned["start_s"].notna()
    if not include_judged:
        keep &= cleaned["verdict"] == ""
    if clip:
        keep &= cleaned["video_id"] == clip
    return list(cleaned.index[keep])


@dataclass
class TextEntry:
    """A line of text typed into the video window (no terminal needed)."""
    prompt: str
    text: str = ""

    def key(self, code: int) -> str | None:
        """Feed one key code. Returns "done" (Enter), "cancel" (Esc) or None (keep typing)."""
        if code in (10, 13):
            return "done"
        if code == 27:
            return "cancel"
        if code in (8, 127):
            self.text = self.text[:-1]
        elif 32 <= code <= 126:
            self.text += chr(code)
        return None


# ---------------------------------------------------------------- window

def draw_entry(view, entry: TextEntry | None, flash: str | None) -> None:
    """Typing box (or a short status message) over the top of the video."""
    import cv2

    w = view.shape[1]
    if entry is not None:
        cv2.rectangle(view, (0, 104), (w, 170), (30, 30, 30), -1)
        _text(view, entry.prompt, (10, 126), YELLOW, 0.5)
        _text(view, entry.text + "_", (10, 156), WHITE, 0.65, 2)
    elif flash:
        cv2.rectangle(view, (0, 104), (w, 136), (30, 30, 30), -1)
        _text(view, flash, (10, 126), YELLOW, 0.5)


def flag_joints_xy(pose: dict | None, flag: str, w: int, h: int,
                   min_conf: float = 0.3) -> list[tuple[int, int]]:
    """Pixel positions of the flagged limb's joints in a w x h frame (pose uses 0-1 coordinates)."""
    if not pose:
        return []
    return [(int(pose[j][0] * w), int(pose[j][1] * h)) for j in FLAG_JOINTS.get(flag, [])
            if j in pose and pose[j][2] >= min_conf]


def _load_poses(video_id: str) -> dict[int, dict[str, tuple[float, float, float]]]:
    """{frame_idx: {joint: (x_norm, y_norm, conf)}} for the clip, or {} if no keypoints."""
    matches = sorted(KEYPOINT_ROOT.glob(f"camera=*/model={MODEL}/{video_id}.parquet"))
    if not matches:
        return {}
    kp = pd.read_parquet(matches[0], columns=["frame_idx", "keypoint", "x_norm", "y_norm",
                                               "confidence"])
    out: dict[int, dict] = {}
    for f, name, x, y, c in kp.itertuples(index=False):
        out.setdefault(int(f), {})[name] = (float(x), float(y), float(c))
    return out


def _pose_at(poses: dict, frame_idx: int, hold: int = 3) -> dict | None:
    for f in range(frame_idx, frame_idx - hold - 1, -1):  # frames skipped by --stride
        if f in poses:
            return poses[f]
    return None


def _text(img, text: str, org: tuple[int, int], colour, size: float, thick: int = 1) -> None:
    import cv2

    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, size, colour, thick, cv2.LINE_AA)


def compose(frame, row, t: float, first_t: float, last_t: float, pos: int, total: int,
            verdict: str, paused: bool, pose: dict | None):
    """The review view: banner on top, video with the flagged limb circled, keys at the bottom."""
    import cv2
    import numpy as np

    from boulder.render.feedback import bone_group
    from boulder.render.overlay import BONES

    h, w = frame.shape[:2]
    on = row.start_s <= t <= row.end_s
    video = frame.copy()
    limb = {"left_arm_bent": "left_arm", "right_arm_bent": "right_arm",
            "left_foot_readjust": "left_leg", "right_foot_readjust": "right_leg"}.get(row.flag)
    if pose:
        pts = {j: (int(x * w), int(y * h)) for j, (x, y, c) in pose.items() if c >= 0.3}
        for a, b in BONES:
            if a in pts and b in pts:
                mine = limb is not None and bone_group(a, b) == limb
                colour = (RED if on else YELLOW) if mine else WHITE
                cv2.line(video, pts[a], pts[b], colour, 4 if mine else 2, cv2.LINE_AA)
    for x, y in flag_joints_xy(pose, row.flag, w, h):
        cv2.circle(video, (x, y), max(14, w // 25), RED if on else YELLOW, 3, cv2.LINE_AA)
    if on:
        cv2.rectangle(video, (0, 0), (w - 1, h - 1), RED, 6)

    top, bottom = 104, 34
    view = np.zeros((h + top + bottom, w, 3), dtype=np.uint8)
    view[top:top + h] = video
    _text(view, TITLES.get(row.flag, row.flag), (10, 34), RED if on else WHITE, 0.85, 2)
    _text(view, f"flag {row.start}-{row.end or row.start}   now {fmt_time(t)}"
                f"{'   PAUSED' if paused else ''}", (10, 62), WHITE, 0.5)
    _text(view, f"{pos + 1}/{total}   your verdict: {verdict}", (10, 84), GREY, 0.5)
    # loop bar: grey = shown before/after, red = the flag, white = now
    span = max(last_t - first_t, 1e-6)

    def bx(sec: float) -> int:
        return int(np.clip((sec - first_t) / span, 0, 1) * (w - 20)) + 10

    cv2.rectangle(view, (10, 92), (w - 10, 100), (70, 70, 70), -1)
    cv2.rectangle(view, (bx(row.start_s), 92), (max(bx(row.end_s), bx(row.start_s) + 2), 100),
                  RED, -1)
    cv2.line(view, (bx(t), 89), (bx(t), 103), WHITE, 2)
    _text(view, "c correct  w wrong  u unsure  s skip  n note  m missed  q quit",
          (10, top + h + 22), GREY, 0.42)
    return view


def review(path: Path = REVIEW_PATH, include_judged: bool = False, clip: str | None = None,
           pad_s: float = 1.0) -> None:
    import cv2

    reviews = read_reviews(path)
    todo = todo_rows(reviews, include_judged, clip)
    if not todo:
        print("nothing to review (run python -m boulder.review.export first?)")
        return
    print(f"{len(todo)} flags to review; click the video window, then use the keys "
          "(c w u s n m space , . r b q). Notes are typed in the video window too.")
    window = "flag review"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    sized = False
    caps: dict[str, cv2.VideoCapture] = {}
    poses: dict[str, dict] = {}
    pos = 0
    try:
        while 0 <= pos < len(todo):
            i = todo[pos]
            cleaned, _ = clean_reviews(reviews)
            row = cleaned.loc[i]
            found = sorted(RAW_DIR.glob(f"{row.video_id}.*"))
            if not found:
                print(f"skip: no clip {RAW_DIR / row.video_id}.mov")
                pos += 1
                continue
            video = found[0]
            cap = caps.get(row.video_id) or caps.setdefault(row.video_id,
                                                           cv2.VideoCapture(str(video)))
            if row.video_id not in poses:
                poses[row.video_id] = _load_poses(row.video_id)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            first = max(0, int((row.start_s - pad_s) * fps))
            last = min(n_frames - 1, int((row.end_s + pad_s) * fps))
            f_idx, paused, frame = first, False, None
            cap.set(cv2.CAP_PROP_POS_FRAMES, first)
            action = None
            entry: TextEntry | None = None
            purpose, missed_t, missed_kind = "", 0.0, ""
            flash, flash_until = None, 0.0
            while action is None:
                if (not paused and entry is None) or frame is None:
                    if f_idx > last:
                        f_idx = first
                        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
                    ok, frame = cap.read()
                    if not ok:
                        f_idx = last + 1
                        continue
                    f_idx += 1
                t = (f_idx - 1) / fps
                if frame.shape[0] > VIEW_HEIGHT:
                    k = VIEW_HEIGHT / frame.shape[0]
                    frame = cv2.resize(frame, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
                verdict = reviews.loc[i, "verdict"] or "-"
                shown = compose(frame, row, t, first / fps, last / fps, pos, len(todo), verdict,
                                paused, _pose_at(poses[row.video_id], f_idx - 1))
                draw_entry(shown, entry, flash if time.monotonic() < flash_until else None)
                if not sized:  # open the window big enough to read
                    vh, vw = shown.shape[:2]
                    k = min(1.0, 900 / vh) if vh > 900 else max(1.0, 700 / vh)
                    cv2.resizeWindow(window, int(vw * k), int(vh * k))
                    sized = True
                cv2.imshow(window, shown)
                wait = 30 if entry is not None or paused else max(1, int(1000 / fps))
                code = cv2.waitKey(wait)
                if code == -1:
                    continue
                code &= 0xFF

                if entry is not None:  # typing mode: keys go into the text
                    result = entry.key(code)
                    if result is None:
                        continue
                    text, entry = entry.text.strip(), None
                    if result == "cancel":
                        continue
                    if purpose == "note" and text:
                        reviews = add_note(reviews, i, text)
                        write_reviews(reviews, path)
                        flash, flash_until = f"note saved: {text}", time.monotonic() + 2
                    elif purpose == "missed_kind":
                        if SHORT_FLAGS.get(text, text) not in FLAG_TYPES:
                            flash, flash_until = f"unknown type '{text}'", time.monotonic() + 2
                            continue
                        missed_kind, purpose = text, "missed_note"
                        entry = TextEntry("note for the missed mistake (optional), Enter")
                    elif purpose == "missed_note":
                        reviews = add_missed(reviews, row.video_id, missed_t, missed_kind, text)
                        write_reviews(reviews, path)
                        flash = (f"added missed {SHORT_FLAGS.get(missed_kind, missed_kind)}"
                                 f" at {fmt_time(missed_t)}")
                        flash_until = time.monotonic() + 2
                    continue

                ch = chr(code)
                if ch in KEY_VERDICTS:
                    reviews = set_verdict(reviews, i, KEY_VERDICTS[ch])
                    write_reviews(reviews, path)
                    action = "next"
                elif ch == "s":
                    action = "next"
                elif ch == "b":
                    action = "back"
                elif ch == "q":
                    action = "quit"
                elif ch == " ":
                    paused = not paused
                elif ch == "r":
                    f_idx = first
                    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
                    frame = None
                elif ch in (",", ".") and paused:
                    f_idx = max(first, f_idx - 2) if ch == "," else f_idx
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                    frame = None
                elif ch == "n":
                    purpose = "note"
                    entry = TextEntry("type a note, Enter saves, Esc cancels")
                elif ch == "m":
                    purpose, missed_t = "missed_kind", t
                    entry = TextEntry(f"missed at {fmt_time(t)}: la ra lf rf p, Enter = other")
            if action == "quit":
                break
            pos += 1 if action == "next" else -1
            pos = max(pos, 0)
    finally:
        for cap in caps.values():
            cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    left = len(todo_rows(reviews, False, clip))
    print(f"saved {path}; {left} flags still without a verdict")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reviews", type=Path, default=REVIEW_PATH)
    parser.add_argument("--all", action="store_true", help="include flags already judged")
    parser.add_argument("--clip", default=None, help="only this video_id")
    parser.add_argument("--pad", type=float, default=1.0, help="seconds shown before and after")
    args = parser.parse_args()
    review(args.reviews, args.all, args.clip, args.pad)


if __name__ == "__main__":
    main()
