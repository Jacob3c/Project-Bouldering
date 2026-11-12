"""Read attempt metadata from a clip's file name.

Naming pattern (gym may contain underscores):
    <date>_<gym>_<colour>-<grade>_att<NN>_<outcome>_<camera>.<ext>
    2026-11-08_SouthSide_Orange-V5-6_att01_send_handheld.MOV
    2026-09-12_Stone_CO_Purple-V3_att01_send_handheld.MOV

Parsed from the right, so underscores in the gym name are fine.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from boulder.ingest.aliases import canonical

_ATTEMPT = re.compile(r"^att(\d+)$", re.IGNORECASE)
_DATE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$")
_GRADE = re.compile(r"^v(b|\d{1,2})(?:-(\d{1,2}))?$", re.IGNORECASE)


@dataclass
class ClipInfo:
    video_id: str
    canonical_id: str | None = None  # video_id with the date zero-padded
    date: str | None = None
    gym: str | None = None
    problem: str | None = None  # e.g. "Orange-V5-6"
    colour: str | None = None   # e.g. "Orange"
    grade: str | None = None    # e.g. "V5-6"
    attempt: int | None = None
    outcome: str | None = None  # "send" | "fell" | other
    camera: str | None = None   # "handheld" | "tripod" | other
    name_ok: bool = False       # True when the name matched the full pattern
    issues: str = ""            # '; '-separated problems, e.g. unknown gym

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_date(text: str) -> str | None:
    """Return the date as YYYY-MM-DD, or None if it is not a real calendar date.

    "2026-1-9" -> "2026-01-09"; "2026-13-45" -> None.
    """
    if not _DATE.match(text):
        return None
    year, month, day = (int(part) for part in text.split("-"))
    try:
        return dt.date(year, month, day).isoformat()
    except ValueError:
        return None


def normalize_grade(text: str) -> str | None:
    """Return a V-grade in one format, or None if it is not a valid V-grade.

    "v3" -> "V3"; "v03-4" -> "V3-4"; "vb" -> "VB"; "V5-3" -> None (range backwards).
    """
    match = _GRADE.match(text.strip())
    if not match:
        return None
    low, high = match.group(1).upper(), match.group(2)
    if low != "B":
        low = str(int(low))
    if high is None:
        return f"V{low}"
    if low == "B" or int(high) <= int(low):
        return None
    return f"V{low}-{int(high)}"


def parse_clip_name(path: str | Path) -> ClipInfo:
    """Best-effort parse; fields stay None when the name doesn't match."""
    video_id = Path(path).stem
    parts = video_id.split("_")
    info = ClipInfo(video_id=video_id)
    if len(parts) < 6:
        return info
    clean_date = normalize_date(parts[0])
    if clean_date is None:
        return info
    match = _ATTEMPT.match(parts[-3])
    if not match:
        return info

    info.date = clean_date
    info.camera = parts[-1].lower()
    info.outcome = parts[-2].lower()
    info.attempt = int(match.group(1))
    info.problem = parts[-4]
    info.gym = "_".join(parts[1:-4])
    colour, _, grade = info.problem.partition("-")
    info.colour = colour or None
    info.grade = grade or None
    info.name_ok = bool(info.gym)
    if not info.name_ok:
        return info

    issues = []
    info.gym, known = canonical("gyms", info.gym)
    if not known:
        issues.append(f"unknown gym '{info.gym}'")
    if info.colour:
        info.colour, known = canonical("colours", info.colour)
        if not known:
            issues.append(f"unknown colour '{info.colour}'")
    if info.grade:
        clean_grade = normalize_grade(info.grade)
        if clean_grade is None:
            issues.append(f"bad grade '{info.grade}'")
        else:
            info.grade = clean_grade
    info.problem = "-".join(p for p in (info.colour, info.grade) if p)
    info.issues = "; ".join(issues)
    info.canonical_id = "_".join([
        info.date, info.gym, info.problem, f"att{info.attempt:02d}",
        info.outcome, info.camera,
    ])
    return info
