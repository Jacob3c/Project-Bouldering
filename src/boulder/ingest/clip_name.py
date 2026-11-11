"""Read attempt metadata from a clip's file name.

Naming pattern (gym may contain underscores):
    <date>_<gym>_<colour>-<grade>_att<NN>_<outcome>_<camera>.<ext>
    2026-11-08_SouthSide_Orange-V5-6_att01_send_handheld.MOV
    2026-09-12_Stone_CO_Purple-V3_att01_send_handheld.MOV

Parsed from the right, so underscores in the gym name are fine.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

_ATTEMPT = re.compile(r"^att(\d+)$", re.IGNORECASE)
_DATE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$")


@dataclass
class ClipInfo:
    video_id: str
    date: str | None = None
    gym: str | None = None
    problem: str | None = None  # e.g. "Orange-V5-6"
    colour: str | None = None   # e.g. "Orange"
    grade: str | None = None    # e.g. "V5-6"
    attempt: int | None = None
    outcome: str | None = None  # "send" | "fell" | other
    camera: str | None = None   # "handheld" | "tripod" | other
    name_ok: bool = False       # True when the name matched the full pattern

    def to_dict(self) -> dict:
        return asdict(self)


def parse_clip_name(path: str | Path) -> ClipInfo:
    """Best-effort parse; fields stay None when the name doesn't match."""
    video_id = Path(path).stem
    parts = video_id.split("_")
    info = ClipInfo(video_id=video_id)
    if len(parts) < 6 or not _DATE.match(parts[0]):
        return info
    match = _ATTEMPT.match(parts[-3])
    if not match:
        return info

    info.date = parts[0]
    info.camera = parts[-1].lower()
    info.outcome = parts[-2].lower()
    info.attempt = int(match.group(1))
    info.problem = parts[-4]
    info.gym = "_".join(parts[1:-4])
    colour, _, grade = info.problem.partition("-")
    info.colour = colour or None
    info.grade = grade or None
    info.name_ok = bool(info.gym)
    return info
