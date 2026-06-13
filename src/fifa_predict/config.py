from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

HISTORICAL_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/"
    "international_results/master/results.csv"
)
OPENFOOTBALL_2026_URL = (
    "https://raw.githubusercontent.com/openfootball/"
    "worldcup/master/2026--usa/cup.txt"
)
PLAYER_RATINGS_DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "justdhia/ea-sports-fc-26-player-ratings"
)
PLAYER_RATINGS_SOURCE = (
    "kaggle:justdhia/ea-sports-fc-26-player-ratings"
)
WORLD_CUP_SQUADS_URL = (
    "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads"
)
FOOTBALL_DATA_WORLD_CUP_URL = (
    "https://api.football-data.org/v4/competitions/WC/matches"
)

HISTORICAL_CACHE = RAW_DIR / "international_results.csv"
FOOTBALL_DATA_CACHE = CACHE_DIR / "football_data_world_cup.json"
OPENFOOTBALL_CACHE = CACHE_DIR / "openfootball_2026.txt"
MODEL_ARTIFACT = ARTIFACT_DIR / "model_bundle.joblib"
METRICS_ARTIFACT = ARTIFACT_DIR / "model_metrics.json"

HOST_TEAMS = frozenset({"Canada", "Mexico", "United States"})


def ensure_directories() -> None:
    for path in (RAW_DIR, CACHE_DIR, PROCESSED_DIR, ARTIFACT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def offline_mode() -> bool:
    return os.getenv("FIFA_PREDICT_OFFLINE", "").lower() in {"1", "true", "yes"}


def api_token() -> str | None:
    return os.getenv("FOOTBALL_DATA_API_TOKEN") or None
