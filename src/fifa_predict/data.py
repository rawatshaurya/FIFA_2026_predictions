from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from fifa_predict import config
from fifa_predict.schemas import utc_now_iso
from fifa_predict.teams import canonical_team

MATCH_COLUMNS = [
    "fixture_id",
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "stage",
    "city",
    "country",
    "neutral",
    "status",
    "source",
    "retrieved_at",
    "score_provenance",
]


class DataUnavailableError(RuntimeError):
    """Raised when neither a remote source nor a local cache is available."""


def _download_text(
    url: str,
    path: Path,
    *,
    session: requests.Session | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> str:
    client = session or requests.Session()
    response = client.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    text = response.content.decode("utf-8-sig")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in MATCH_COLUMNS:
        if column not in output:
            output[column] = pd.NA
    output = output[MATCH_COLUMNS]
    output["date"] = pd.to_datetime(output["date"], utc=True, errors="coerce")
    output["home_team"] = output["home_team"].map(canonical_team)
    output["away_team"] = output["away_team"].map(canonical_team)
    output["home_score"] = pd.to_numeric(output["home_score"], errors="coerce").astype(
        "Int64"
    )
    output["away_score"] = pd.to_numeric(output["away_score"], errors="coerce").astype(
        "Int64"
    )
    output["neutral"] = output["neutral"].fillna(False).astype(bool)
    output["status"] = output["status"].fillna("SCHEDULED")
    return deduplicate_matches(output)


def deduplicate_matches(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate source records while preferring rows with scores."""
    output = frame.copy()
    output["_completed"] = (
        output["home_score"].notna() & output["away_score"].notna()
    ).astype(int)
    output["_retrieved"] = pd.to_datetime(
        output["retrieved_at"], utc=True, errors="coerce"
    )
    output["_day"] = pd.to_datetime(output["date"], utc=True).dt.floor("D")
    output = output.sort_values(["_completed", "_retrieved"], na_position="first")
    output = output.drop_duplicates(
        subset=["_day", "home_team", "away_team", "tournament"], keep="last"
    )
    return (
        output.drop(columns=["_completed", "_retrieved", "_day"])
        .sort_values(["date", "home_team", "away_team"])
        .reset_index(drop=True)
    )


def load_international_results(
    *,
    cache_path: Path = config.HISTORICAL_CACHE,
    refresh: bool = False,
    offline: bool | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Load the CC0 international results data and normalize its schema."""
    config.ensure_directories()
    is_offline = config.offline_mode() if offline is None else offline
    if (refresh or not cache_path.exists()) and not is_offline:
        try:
            _download_text(
                config.HISTORICAL_RESULTS_URL, cache_path, session=session
            )
        except requests.RequestException:
            if not cache_path.exists():
                raise DataUnavailableError(
                    "Historical results download failed and no cache exists."
                ) from None
    if not cache_path.exists():
        raise DataUnavailableError(
            f"Historical results cache not found at {cache_path}."
        )

    source = pd.read_csv(cache_path)
    required = {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "city",
        "country",
        "neutral",
    }
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Historical data is missing columns: {sorted(missing)}")

    retrieved_at = datetime.fromtimestamp(
        cache_path.stat().st_mtime, tz=timezone.utc
    ).isoformat()
    frame = source.assign(
        fixture_id=pd.NA,
        stage=source["tournament"],
        status="FINISHED",
        source="martj42/international_results",
        retrieved_at=retrieved_at,
        score_provenance="published final score",
    )
    frame = _normalize_frame(frame)
    cutoff = pd.Timestamp("2000-01-01", tz="UTC")
    completed = frame["home_score"].notna() & frame["away_score"].notna()
    return frame.loc[completed & (frame["date"] >= cutoff)].reset_index(drop=True)


def parse_football_data(payload: dict[str, Any], retrieved_at: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for match in payload.get("matches", []):
        score = match.get("score") or {}
        regular = score.get("regularTime") or {}
        full_time = score.get("fullTime") or {}
        home_score = regular.get("home")
        away_score = regular.get("away")
        provenance = "regularTime"
        if home_score is None and match.get("status") == "FINISHED":
            home_score = full_time.get("home")
            away_score = full_time.get("away")
            provenance = "fullTime fallback"
        rows.append(
            {
                "fixture_id": str(match.get("id")) if match.get("id") else pd.NA,
                "date": match.get("utcDate"),
                "home_team": (match.get("homeTeam") or {}).get("name"),
                "away_team": (match.get("awayTeam") or {}).get("name"),
                "home_score": home_score,
                "away_score": away_score,
                "tournament": "FIFA World Cup",
                "stage": match.get("stage") or "UNKNOWN",
                "city": pd.NA,
                "country": pd.NA,
                "neutral": True,
                "status": match.get("status") or "SCHEDULED",
                "source": "football-data.org",
                "retrieved_at": retrieved_at,
                "score_provenance": provenance if home_score is not None else pd.NA,
            }
        )
    if not rows:
        raise ValueError("football-data.org payload contains no matches")
    return _normalize_frame(pd.DataFrame(rows))


_DATE_RE = re.compile(
    r"^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+"
    r"(?P<month>June|Jun|July|Jul)\s+(?P<day>\d{1,2})$",
    re.IGNORECASE,
)
_MATCH_RE = re.compile(
    r"^\s*(?P<time>\d{1,2}:\d{2})\s+UTC(?P<offset>[+-]\d{1,2})\s+"
    r"(?P<home>.+?)\s+"
    r"(?:(?P<home_score>\d+)-(?P<away_score>\d+)"
    r"(?:\s+(?:a\.e\.t\.\s+)?\([^)]*\))?|v)\s+"
    r"(?P<away>.+?)\s+@\s+(?P<venue>.+?)\s*$",
    re.IGNORECASE,
)


def parse_openfootball(text: str, retrieved_at: str) -> pd.DataFrame:
    """Parse the public Football.TXT 2026 schedule and completed scores."""
    rows: list[dict[str, Any]] = []
    current_day: int | None = None
    current_month: int | None = None
    current_stage = "UNKNOWN"

    for raw_line in text.splitlines():
        line = raw_line.strip()
        group_match = re.match(r"^▪\s+Group\s+([A-L])$", line)
        if group_match:
            current_stage = f"GROUP_{group_match.group(1)}"
            continue
        stage_match = re.match(
            r"^▪\s+(Round of 32|Round of 16|Quarter-finals?|"
            r"Semi-finals?|Match for third place|Final)$",
            line,
            re.IGNORECASE,
        )
        if stage_match:
            current_stage = stage_match.group(1).upper().replace(" ", "_")
            continue
        date_match = _DATE_RE.match(line)
        if date_match:
            current_day = int(date_match.group("day"))
            current_month = (
                6 if date_match.group("month").casefold().startswith("jun") else 7
            )
            continue
        match = _MATCH_RE.match(raw_line)
        if not match or current_day is None or current_month is None:
            continue

        hour, minute = (int(value) for value in match.group("time").split(":"))
        offset_hours = int(match.group("offset"))
        local = datetime(
            2026,
            current_month,
            current_day,
            hour,
            minute,
            tzinfo=timezone(timedelta(hours=offset_hours)),
        )
        home_score = match.group("home_score")
        away_score = match.group("away_score")
        completed = home_score is not None and away_score is not None
        home = canonical_team(match.group("home"))
        away = canonical_team(match.group("away"))
        rows.append(
            {
                "fixture_id": pd.NA,
                "date": local.astimezone(timezone.utc),
                "home_team": home,
                "away_team": away,
                "home_score": home_score,
                "away_score": away_score,
                "tournament": "FIFA World Cup",
                "stage": current_stage,
                "city": match.group("venue"),
                "country": pd.NA,
                "neutral": True,
                "status": "FINISHED" if completed else "SCHEDULED",
                "source": "openfootball/worldcup",
                "retrieved_at": retrieved_at,
                "score_provenance": "Football.TXT score" if completed else pd.NA,
            }
        )
    if not rows:
        raise ValueError("No World Cup matches could be parsed from Football.TXT")
    return _normalize_frame(pd.DataFrame(rows))


def _read_cached_api(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    retrieved_at = payload.pop("_retrieved_at", None) or utc_now_iso()
    return parse_football_data(payload, retrieved_at)


def load_world_cup_matches(
    *,
    token: str | None = None,
    refresh: bool = True,
    offline: bool | None = None,
    session: requests.Session | None = None,
    api_cache: Path = config.FOOTBALL_DATA_CACHE,
    public_cache: Path = config.OPENFOOTBALL_CACHE,
) -> pd.DataFrame:
    """Load 2026 fixtures with API -> cache -> public snapshot fallback."""
    config.ensure_directories()
    is_offline = config.offline_mode() if offline is None else offline
    token = token if token is not None else config.api_token()
    client = session or requests.Session()

    if token and refresh and not is_offline:
        try:
            response = client.get(
                config.FOOTBALL_DATA_WORLD_CUP_URL,
                headers={"X-Auth-Token": token},
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            retrieved_at = utc_now_iso()
            cached_payload = dict(payload)
            cached_payload["_retrieved_at"] = retrieved_at
            api_cache.parent.mkdir(parents=True, exist_ok=True)
            api_cache.write_text(
                json.dumps(cached_payload, indent=2), encoding="utf-8"
            )
            return parse_football_data(payload, retrieved_at)
        except (requests.RequestException, ValueError, json.JSONDecodeError):
            pass

    if api_cache.exists() and (is_offline or not refresh):
        try:
            return _read_cached_api(api_cache)
        except (ValueError, json.JSONDecodeError, KeyError):
            pass

    if (refresh or not public_cache.exists()) and not is_offline:
        try:
            text = _download_text(
                config.OPENFOOTBALL_2026_URL,
                public_cache,
                session=client,
            )
            return parse_openfootball(text, utc_now_iso())
        except (requests.RequestException, ValueError):
            pass

    if api_cache.exists():
        try:
            return _read_cached_api(api_cache)
        except (ValueError, json.JSONDecodeError, KeyError):
            pass

    if public_cache.exists():
        retrieved_at = datetime.fromtimestamp(
            public_cache.stat().st_mtime, tz=timezone.utc
        ).isoformat()
        return parse_openfootball(
            public_cache.read_text(encoding="utf-8"), retrieved_at
        )

    raise DataUnavailableError(
        "World Cup data unavailable: API, API cache, and public snapshot failed."
    )


def combine_completed_matches(
    historical: pd.DataFrame, world_cup: pd.DataFrame
) -> pd.DataFrame:
    completed = world_cup.loc[
        world_cup["home_score"].notna() & world_cup["away_score"].notna()
    ].copy()
    retained_historical = historical.copy()
    # The historical CSV stores dates without kickoff times, while live feeds
    # use UTC timestamps. A late local kickoff can therefore land on the next
    # UTC day; remove matching live records within one day before concatenating.
    for row in completed.to_dict("records"):
        same_match = (
            (retained_historical["home_team"] == row["home_team"])
            & (retained_historical["away_team"] == row["away_team"])
            & (retained_historical["tournament"] == row["tournament"])
            & (
                (
                    pd.to_datetime(retained_historical["date"], utc=True).dt.floor(
                        "D"
                    )
                    - pd.to_datetime(row["date"], utc=True).floor("D")
                )
                .abs()
                .dt.days
                <= 1
            )
        )
        retained_historical = retained_historical.loc[~same_match]
    return deduplicate_matches(
        pd.concat([retained_historical, completed], ignore_index=True)
    )
