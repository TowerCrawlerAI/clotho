"""Slugify FML headings into snake_case EntityIds.

See `docs/design/LFR.md` § EntityId.
"""

from __future__ import annotations

import re

_APOSTROPHES = re.compile(r"['’]")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_MULTI_UNDERSCORE = re.compile(r"_+")


def slugify(heading: str) -> str:
    """Lowercase ASCII snake_case slug.

    Apostrophes are dropped before substitution so `King's` becomes `kings`
    rather than `king_s`. Other punctuation is replaced with underscores.
    """
    s = heading.lower()
    s = _APOSTROPHES.sub("", s)
    s = _NON_ALNUM.sub("_", s)
    s = _MULTI_UNDERSCORE.sub("_", s).strip("_")
    return s
