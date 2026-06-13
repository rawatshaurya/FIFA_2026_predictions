from __future__ import annotations

import re
import unicodedata

TEAM_ALIASES = {
    "bosnia & herzegovina": "Bosnia and Herzegovina",
    "bosnia-herzegovina": "Bosnia and Herzegovina",
    "cape verde islands": "Cape Verde",
    "cabo verde": "Cape Verde",
    "china pr": "China",
    "congo dr": "DR Congo",
    "côte d'ivoire": "Ivory Coast",
    "cote d'ivoire": "Ivory Coast",
    "czechia": "Czech Republic",
    "dpr korea": "North Korea",
    "england": "England",
    "holland": "Netherlands",
    "ir iran": "Iran",
    "korea republic": "South Korea",
    "kyrgyz republic": "Kyrgyzstan",
    "north macedonia": "North Macedonia",
    "republic of ireland": "Republic of Ireland",
    "russia": "Russia",
    "slovak republic": "Slovakia",
    "türkiye": "Turkey",
    "turkiye": "Turkey",
    "u.s.a.": "United States",
    "united states of america": "United States",
    "usa": "United States",
}


def _key(name: str) -> str:
    value = unicodedata.normalize("NFKC", str(name)).strip()
    value = re.sub(r"\s+", " ", value)
    return value.casefold()


def canonical_team(name: str) -> str:
    """Return a stable display name shared by all data sources."""
    cleaned = unicodedata.normalize("NFKC", str(name)).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        raise ValueError("Team name cannot be empty")
    return TEAM_ALIASES.get(_key(cleaned), cleaned)
