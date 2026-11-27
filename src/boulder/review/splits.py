"""Which clips are for tuning and which are kept back for testing.

A rule tuned and checked on the same clips looks better than it really is. So about
1 clip in 4 is a "test" clip: tools tune on the other clips only, then report how the
chosen setting does on the test clips it never saw.

The split is fixed by the clip's ID (a hash), so a clip never changes sides and new
clips are assigned automatically. To move a clip by hand, add it to
metadata/splits.csv with columns clip_id,split (split = tune or test).
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

SPLITS_PATH = Path("metadata/splits.csv")
TEST_SHARE = 4  # 1 clip in this many goes to test


def default_split(clip_id: str) -> str:
    """'test' for about 1 clip in TEST_SHARE, else 'tune'; never changes for a clip."""
    h = int(hashlib.sha1(clip_id.encode()).hexdigest(), 16)
    return "test" if h % TEST_SHARE == 0 else "tune"


def load_overrides(path: Path = SPLITS_PATH) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {r["clip_id"].strip(): r["split"].strip().lower() for r in csv.DictReader(f)
                if r.get("clip_id") and r.get("split", "").strip().lower() in ("tune", "test")}


def split_of(clip_id: str, overrides: dict[str, str] | None = None) -> str:
    return (overrides or {}).get(clip_id, default_split(clip_id))
