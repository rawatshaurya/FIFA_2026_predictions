from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class PredictionRecord:
    home_team: str
    away_team: str
    match_date: str
    stage: str
    neutral: bool
    home_win_probability: float
    draw_probability: float
    away_win_probability: float
    predicted_result: str
    eventual_home_probability: float | None
    eventual_away_probability: float | None
    model_version: str
    as_of: str
    fixture_id: str | None = None
    source: str | None = None
    fixture_status: str | None = None
    actual_home_score: int | None = None
    actual_away_score: int | None = None
    actual_result: str | None = None
    prediction_correct: bool | None = None
    player_context_applied: bool = False
    home_expected_xi_quality: float | None = None
    away_expected_xi_quality: float | None = None
    home_squad_depth_quality: float | None = None
    away_squad_depth_quality: float | None = None
    home_unavailable_quality: float | None = None
    away_unavailable_quality: float | None = None
    home_availability_rate: float | None = None
    away_availability_rate: float | None = None
    home_lineup_confirmed: bool | None = None
    away_lineup_confirmed: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
