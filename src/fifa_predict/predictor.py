from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fifa_predict.features import FEATURE_COLUMNS, regulation_result
from fifa_predict.models import CLASSES, ModelBundle
from fifa_predict.players import PlayerContextStore, adjust_probabilities
from fifa_predict.schemas import PredictionRecord, utc_now_iso
from fifa_predict.teams import canonical_team


def is_knockout_stage(stage: str) -> bool:
    value = str(stage).casefold().replace("_", " ")
    return not any(term in value for term in ("group", "league", "qualification")) and any(
        term in value
        for term in (
            "round of",
            "last 32",
            "last 16",
            "quarter",
            "semi",
            "third place",
            "final",
        )
    )


class MatchPredictor:
    def __init__(
        self,
        bundle: ModelBundle,
        *,
        as_of: str | None = None,
        player_context: PlayerContextStore | None = None,
    ) -> None:
        self.bundle = bundle
        self.feature_engine = bundle.feature_engine.copy()
        self.as_of = as_of or utc_now_iso()
        self.player_context = player_context or PlayerContextStore()

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        player_context: PlayerContextStore | None = None,
    ) -> "MatchPredictor":
        return cls(
            joblib.load(path),
            player_context=player_context or PlayerContextStore.from_csv(),
        )

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        match_date: pd.Timestamp | str,
        stage: str,
        neutral: bool,
        *,
        fixture_id: str | None = None,
        source: str | None = None,
    ) -> PredictionRecord:
        home = canonical_team(home_team)
        away = canonical_team(away_team)
        features = self.feature_engine.make_features(
            home,
            away,
            match_date,
            tournament="FIFA World Cup",
            stage=stage,
            neutral=neutral,
        )
        feature_frame = pd.DataFrame([features], columns=FEATURE_COLUMNS)
        probabilities = self.bundle.predict_proba(feature_frame)[0]
        player_context = self.player_context.match_context(
            home,
            away,
            match_date,
            fixture_id=fixture_id,
            as_of=self.as_of,
        )
        probabilities = adjust_probabilities(probabilities, player_context)
        player_context_applied = (
            player_context.home.player_data_available
            and player_context.away.player_data_available
        )
        predicted = str(CLASSES[int(np.argmax(probabilities))])
        result_names = {"H": "home_win", "D": "draw", "A": "away_win"}

        eventual_home: float | None = None
        eventual_away: float | None = None
        if is_knockout_stage(stage):
            elo_difference = features["elo_diff"]
            strength_share = 1.0 / (1.0 + 10 ** (-elo_difference / 400.0))
            # Extra time and penalties add substantial randomness, so shrink
            # the Elo-derived split halfway toward an even contest.
            draw_home_share = 0.5 + 0.5 * (strength_share - 0.5)
            eventual_home = float(
                probabilities[0] + probabilities[1] * draw_home_share
            )
            eventual_away = float(
                probabilities[2] + probabilities[1] * (1 - draw_home_share)
            )

        return PredictionRecord(
            home_team=home,
            away_team=away,
            match_date=pd.to_datetime(match_date, utc=True).isoformat(),
            stage=stage,
            neutral=bool(neutral),
            home_win_probability=float(probabilities[0]),
            draw_probability=float(probabilities[1]),
            away_win_probability=float(probabilities[2]),
            predicted_result=result_names[predicted],
            eventual_home_probability=eventual_home,
            eventual_away_probability=eventual_away,
            model_version=self.bundle.model_version,
            as_of=self.as_of,
            fixture_id=fixture_id,
            source=source,
            player_context_applied=player_context_applied,
            home_expected_xi_quality=(
                player_context.home.expected_xi_quality
                if player_context.home.player_data_available
                else None
            ),
            away_expected_xi_quality=(
                player_context.away.expected_xi_quality
                if player_context.away.player_data_available
                else None
            ),
            home_squad_depth_quality=(
                player_context.home.squad_depth_quality
                if player_context.home.player_data_available
                else None
            ),
            away_squad_depth_quality=(
                player_context.away.squad_depth_quality
                if player_context.away.player_data_available
                else None
            ),
            home_unavailable_quality=(
                player_context.home.unavailable_quality
                if player_context.home.player_data_available
                else None
            ),
            away_unavailable_quality=(
                player_context.away.unavailable_quality
                if player_context.away.player_data_available
                else None
            ),
            home_availability_rate=(
                player_context.home.availability_rate
                if player_context.home.player_data_available
                else None
            ),
            away_availability_rate=(
                player_context.away.availability_rate
                if player_context.away.player_data_available
                else None
            ),
            home_lineup_confirmed=(
                player_context.home.lineup_confirmed
                if player_context.home.player_data_available
                else None
            ),
            away_lineup_confirmed=(
                player_context.away.lineup_confirmed
                if player_context.away.player_data_available
                else None
            ),
        )

    def predict_fixtures(self, fixtures: pd.DataFrame) -> list[PredictionRecord]:
        records: list[PredictionRecord] = []
        for row in fixtures.sort_values("date").to_dict("records"):
            fixture_id = row.get("fixture_id")
            if pd.isna(fixture_id):
                fixture_id = None
            records.append(
                self.predict_match(
                    str(row["home_team"]),
                    str(row["away_team"]),
                    row["date"],
                    str(row["stage"]),
                    bool(row["neutral"]),
                    fixture_id=str(fixture_id) if fixture_id is not None else None,
                    source=str(row.get("source")) if row.get("source") else None,
                )
            )
        return records

    def predict_tournament_timeline(
        self, fixtures: pd.DataFrame
    ) -> list[PredictionRecord]:
        """Predict all fixtures and roll state forward only after each match day."""
        timeline = fixtures.copy()
        timeline["date"] = pd.to_datetime(timeline["date"], utc=True)
        records: list[PredictionRecord] = []

        for _, day_fixtures in timeline.sort_values("date").groupby(
            timeline["date"].dt.floor("D")
        ):
            day_records = self.predict_fixtures(day_fixtures)
            rows = day_fixtures.sort_values("date").to_dict("records")
            for prediction, row in zip(day_records, rows):
                completed = pd.notna(row["home_score"]) and pd.notna(
                    row["away_score"]
                )
                actual = None
                correct = None
                home_score = None
                away_score = None
                if completed:
                    home_score = int(row["home_score"])
                    away_score = int(row["away_score"])
                    code = regulation_result(home_score, away_score)
                    actual = {"H": "home_win", "D": "draw", "A": "away_win"}[code]
                    correct = prediction.predicted_result == actual
                records.append(
                    replace(
                        prediction,
                        fixture_status=str(row.get("status") or "SCHEDULED"),
                        actual_home_score=home_score,
                        actual_away_score=away_score,
                        actual_result=actual,
                        prediction_correct=correct,
                    )
                )
            self.feature_engine.update_completed_matches(day_fixtures)
        return records


def export_predictions(
    predictions: list[PredictionRecord],
    output_directory: Path,
    *,
    stem: str,
) -> tuple[Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    rows = [prediction.to_dict() for prediction in predictions]
    csv_path = output_directory / f"{stem}.csv"
    json_path = output_directory / f"{stem}.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return csv_path, json_path
