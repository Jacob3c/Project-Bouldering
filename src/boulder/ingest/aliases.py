"""Map different spellings of gyms, tag colours and wall styles to one official name.

The official names live in configs/aliases.yaml. Lookups ignore case and
separators (spaces, hyphens, underscores), so only spellings that differ in
letters need listing there.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml

# aliases.py -> ingest/ -> boulder/ -> src/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
ALIASES_PATH = REPO_ROOT / "configs" / "aliases.yaml"

_NOT_ALNUM = re.compile(r"[^a-z0-9]")


def _key(text: str) -> str:
    """Lower-case and drop every non-letter, non-digit: 'Main-Rec' -> 'mainrec'."""
    return _NOT_ALNUM.sub("", text.lower())


@lru_cache(maxsize=None)
def _lookup(kind: str) -> dict[str, str]:
    """Build {key: official name} for one section of the alias file (read once)."""
    with open(ALIASES_PATH) as f:
        section = yaml.safe_load(f).get(kind) or {}
    table: dict[str, str] = {}
    for official, variants in section.items():
        for name in [official, *(variants or [])]:
            table[_key(str(name))] = str(official)
    return table


def canonical(kind: str, raw: str) -> tuple[str, bool]:
    """Return (official name, True) if known, else (raw stripped, False).

    kind is a section of aliases.yaml: "gyms", "colours" or "wall_styles".
    """
    official = _lookup(kind).get(_key(raw))
    if official is None:
        return raw.strip(), False
    return official, True
