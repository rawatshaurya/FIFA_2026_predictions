from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from fifa_predict import config
from fifa_predict.data import (
    combine_completed_matches,
    load_international_results,
    load_world_cup_matches,
)
from fifa_predict.features import FeatureEngine
from fifa_predict.models import ModelBundle, train_and_select
from fifa_predict.players import PlayerContextStore
from fifa_predict.predictor import MatchPredictor, export_predictions
from fifa_predict.schemas import PredictionRecord, utc_now_iso


@dataclass
class DataAudit:
    historical: pd.DataFrame
    world_cup: pd.DataFrame

    def summary(self) -> dict[str, Any]:
        completed = self.world_cup["home_score"].notna()
        return {
            "historical_matches": len(self.historical),
            "historical_start": self.historical["date"].min().isoformat(),
            "historical_end": self.historical["date"].max().isoformat(),
            "world_cup_fixtures": len(self.world_cup),
            "world_cup_completed": int(completed.sum()),
            "world_cup_remaining": int((~completed).sum()),
            "world_cup_source": sorted(self.world_cup["source"].dropna().unique()),
            "retrieved_at": sorted(
                self.world_cup["retrieved_at"].dropna().astype(str).unique()
            )[-1],
        }


def audit_sources(
    *, refresh: bool = True, offline: bool | None = None
) -> DataAudit:
    historical = load_international_results(refresh=refresh, offline=offline)
    world_cup = load_world_cup_matches(refresh=refresh, offline=offline)
    return DataAudit(historical=historical, world_cup=world_cup)


def fit_model(
    matches: pd.DataFrame,
    *,
    save: bool = True,
    n_splits: int = 5,
) -> tuple[ModelBundle, pd.DataFrame]:
    engine = FeatureEngine()
    X, y, audit = engine.build_training_set(matches)
    bundle = train_and_select(
        X,
        y,
        engine,
        audit["date"],
        n_splits=n_splits,
    )
    if save:
        config.ensure_directories()
        joblib.dump(bundle, config.MODEL_ARTIFACT)
        config.METRICS_ARTIFACT.write_text(
            json.dumps(bundle.metrics, indent=2), encoding="utf-8"
        )
    return bundle, audit


def train_from_sources(
    *, refresh: bool = True, offline: bool | None = None, save: bool = True
) -> tuple[ModelBundle, DataAudit]:
    audit = audit_sources(refresh=refresh, offline=offline)
    matches = combine_completed_matches(audit.historical, audit.world_cup)
    bundle, _ = fit_model(matches, save=save)
    return bundle, audit


def predict_remaining(
    *,
    refresh: bool = True,
    offline: bool | None = None,
    output_directory: Path = config.ARTIFACT_DIR,
    now: pd.Timestamp | str | None = None,
) -> tuple[list[PredictionRecord], tuple[Path, Path], ModelBundle, DataAudit]:
    bundle, audit = train_from_sources(
        refresh=refresh, offline=offline, save=True
    )
    as_of_timestamp = (
        pd.Timestamp.now(tz="UTC")
        if now is None
        else pd.to_datetime(now, utc=True)
    )
    remaining = audit.world_cup.loc[
        audit.world_cup["home_score"].isna()
        & audit.world_cup["away_score"].isna()
        & (audit.world_cup["date"] >= as_of_timestamp)
    ].copy()
    predictor = MatchPredictor(
        bundle,
        as_of=as_of_timestamp.isoformat(),
        player_context=PlayerContextStore.from_csv(),
    )
    predictions = predictor.predict_fixtures(remaining)
    stem = f"predictions_{as_of_timestamp.strftime('%Y%m%d_%H%M%S')}"
    paths = export_predictions(predictions, output_directory, stem=stem)
    return predictions, paths, bundle, audit


def predict_tournament(
    *,
    refresh: bool = True,
    offline: bool | None = None,
    output_directory: Path = config.ARTIFACT_DIR,
    now: pd.Timestamp | str | None = None,
) -> tuple[list[PredictionRecord], tuple[Path, Path], ModelBundle, DataAudit]:
    """Backtest completed fixtures and predict scheduled fixtures without leakage."""
    audit = audit_sources(refresh=refresh, offline=offline)
    tournament_start = audit.world_cup["date"].min().floor("D")
    pre_tournament = audit.historical.loc[
        audit.historical["date"].dt.floor("D") < tournament_start
    ].copy()
    bundle, _ = fit_model(pre_tournament, save=True)
    as_of_timestamp = (
        pd.Timestamp.now(tz="UTC")
        if now is None
        else pd.to_datetime(now, utc=True)
    )
    predictor = MatchPredictor(
        bundle,
        as_of=as_of_timestamp.isoformat(),
        player_context=PlayerContextStore.from_csv(),
    )
    predictions = predictor.predict_tournament_timeline(audit.world_cup)
    stem = f"tournament_predictions_{as_of_timestamp.strftime('%Y%m%d_%H%M%S')}"
    paths = export_predictions(predictions, output_directory, stem=stem)
    return predictions, paths, bundle, audit


def predict_match(
    home_team: str,
    away_team: str,
    match_date: pd.Timestamp | str,
    stage: str,
    neutral: bool,
    *,
    model_path: Path = config.MODEL_ARTIFACT,
) -> PredictionRecord:
    """Public convenience interface backed by the persisted model artifact."""
    predictor = MatchPredictor(
        joblib.load(model_path),
        player_context=PlayerContextStore.from_csv(),
    )
    return predictor.predict_match(
        home_team, away_team, match_date, stage, neutral
    )
