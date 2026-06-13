from pathlib import Path

import pandas as pd
import pytest
import requests

from fifa_predict.data import (
    combine_completed_matches,
    deduplicate_matches,
    load_world_cup_matches,
    parse_football_data,
    parse_openfootball,
)

OPENFOOTBALL_SAMPLE = """= World Cup 2026
▪ Group A
Thu June 11
  13:00 UTC-6  Mexico  2-0 (1-0)  South Africa  @ Mexico City
  20:00 UTC-6  South Korea  v Czech Republic  @ Guadalajara (Zapopan)
▪ Round of 32
Sun June 28
  15:00 UTC-4  Mexico v Canada @ Miami (Miami Gardens)
"""


def test_parse_openfootball_scores_and_schedule() -> None:
    frame = parse_openfootball(OPENFOOTBALL_SAMPLE, "2026-06-13T12:00:00+00:00")
    assert len(frame) == 3
    assert frame.iloc[0]["home_score"] == 2
    assert frame.iloc[0]["away_score"] == 0
    assert frame.iloc[1]["status"] == "SCHEDULED"
    assert frame.iloc[2]["stage"] == "ROUND_OF_32"
    assert frame.iloc[0]["date"] == pd.Timestamp("2026-06-11T19:00:00Z")


def test_parse_football_data_uses_regulation_score() -> None:
    payload = {
        "matches": [
            {
                "id": 7,
                "utcDate": "2026-07-01T20:00:00Z",
                "status": "FINISHED",
                "stage": "LAST_32",
                "homeTeam": {"name": "USA"},
                "awayTeam": {"name": "Türkiye"},
                "score": {
                    "regularTime": {"home": 1, "away": 1},
                    "fullTime": {"home": 2, "away": 1},
                },
            }
        ]
    }
    frame = parse_football_data(payload, "2026-07-01T23:00:00Z")
    assert frame.iloc[0]["home_team"] == "United States"
    assert frame.iloc[0]["away_team"] == "Turkey"
    assert frame.iloc[0]["home_score"] == 1
    assert frame.iloc[0]["away_score"] == 1
    assert frame.iloc[0]["score_provenance"] == "regularTime"


class FailingSession:
    def get(self, *args, **kwargs):
        raise requests.ConnectionError("offline")


def test_api_failure_falls_back_to_public_cache(tmp_path: Path) -> None:
    public_cache = tmp_path / "worldcup.txt"
    public_cache.write_text(OPENFOOTBALL_SAMPLE, encoding="utf-8")
    frame = load_world_cup_matches(
        token="bad-token",
        refresh=True,
        offline=False,
        session=FailingSession(),
        api_cache=tmp_path / "missing.json",
        public_cache=public_cache,
    )
    assert len(frame) == 3
    assert set(frame["source"]) == {"openfootball/worldcup"}


def test_missing_token_uses_cache_without_network(tmp_path: Path) -> None:
    public_cache = tmp_path / "worldcup.txt"
    public_cache.write_text(OPENFOOTBALL_SAMPLE, encoding="utf-8")
    frame = load_world_cup_matches(
        token=None,
        refresh=False,
        offline=True,
        public_cache=public_cache,
        api_cache=tmp_path / "missing.json",
    )
    assert len(frame) == 3


def test_deduplication_prefers_completed_record() -> None:
    rows = pd.DataFrame(
        [
            {
                "date": "2026-06-11T19:00:00Z",
                "home_team": "Mexico",
                "away_team": "South Africa",
                "tournament": "FIFA World Cup",
                "home_score": pd.NA,
                "away_score": pd.NA,
                "retrieved_at": "2026-06-10T00:00:00Z",
            },
            {
                "date": "2026-06-11T19:00:00Z",
                "home_team": "Mexico",
                "away_team": "South Africa",
                "tournament": "FIFA World Cup",
                "home_score": 2,
                "away_score": 0,
                "retrieved_at": "2026-06-12T00:00:00Z",
            },
        ]
    )
    output = deduplicate_matches(rows)
    assert len(output) == 1
    assert output.iloc[0]["home_score"] == 2


def test_live_utc_date_replaces_date_only_historical_record() -> None:
    base = {
        "fixture_id": pd.NA,
        "home_team": "United States",
        "away_team": "Paraguay",
        "home_score": 1,
        "away_score": 0,
        "tournament": "FIFA World Cup",
        "stage": "GROUP_D",
        "city": "Inglewood",
        "country": "United States",
        "neutral": False,
        "status": "FINISHED",
        "source": "test",
        "retrieved_at": "2026-06-13T02:00:00Z",
        "score_provenance": "test",
    }
    historical = pd.DataFrame([{**base, "date": pd.Timestamp("2026-06-12", tz="UTC")}])
    live = pd.DataFrame(
        [{**base, "date": pd.Timestamp("2026-06-13T01:00:00Z"), "source": "live"}]
    )
    combined = combine_completed_matches(historical, live)
    assert len(combined) == 1
    assert combined.iloc[0]["source"] == "live"
