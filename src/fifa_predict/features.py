from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from fifa_predict.config import HOST_TEAMS
from fifa_predict.teams import canonical_team

DEFAULT_ELO = 1500.0
HOME_ELO_ADVANTAGE = 60.0
DEFAULT_GOALS = 1.2

FEATURE_COLUMNS = [
    "elo_home",
    "elo_away",
    "elo_diff",
    "home_form_5",
    "away_form_5",
    "form_5_diff",
    "home_form_10",
    "away_form_10",
    "form_10_diff",
    "home_goals_for_5",
    "away_goals_for_5",
    "goals_for_5_diff",
    "home_goals_against_5",
    "away_goals_against_5",
    "goals_against_5_diff",
    "home_win_rate_10",
    "away_win_rate_10",
    "home_draw_rate_10",
    "away_draw_rate_10",
    "home_loss_rate_10",
    "away_loss_rate_10",
    "home_rest_days",
    "away_rest_days",
    "rest_days_diff",
    "neutral",
    "home_advantage",
    "home_is_host",
    "away_is_host",
    "competition_importance",
]


@dataclass
class TeamState:
    elo: float = DEFAULT_ELO
    outcomes: deque[str] = field(default_factory=lambda: deque(maxlen=10))
    goals_for: deque[int] = field(default_factory=lambda: deque(maxlen=10))
    goals_against: deque[int] = field(default_factory=lambda: deque(maxlen=10))
    last_match_date: pd.Timestamp | None = None
    matches: int = 0


def competition_importance(tournament: str, stage: str = "") -> float:
    value = f"{tournament} {stage}".casefold()
    if "fifa world cup" in value and "qualification" not in value:
        return 4.0
    if any(
        name in value
        for name in (
            "uefa euro",
            "copa américa",
            "copa america",
            "african cup of nations",
            "asian cup",
            "gold cup",
        )
    ) and "qualification" not in value:
        return 3.25
    if "qualification" in value or "qualifier" in value:
        return 2.5
    if "nations league" in value:
        return 1.75
    if "friendly" in value:
        return 1.0
    return 1.5


def regulation_result(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def _points(outcomes: Iterable[str], window: int) -> float:
    values = list(outcomes)[-window:]
    if not values:
        return 0.5
    points = sum(3 if item == "W" else 1 if item == "D" else 0 for item in values)
    return points / (3 * len(values))


def _rate(outcomes: Iterable[str], result: str, window: int = 10) -> float:
    values = list(outcomes)[-window:]
    if not values:
        return 1 / 3
    return values.count(result) / len(values)


def _mean(values: Iterable[int], window: int) -> float:
    selected = list(values)[-window:]
    return float(np.mean(selected)) if selected else DEFAULT_GOALS


def _rest_days(state: TeamState, match_date: pd.Timestamp) -> float:
    if state.last_match_date is None:
        return 30.0
    days = (match_date.floor("D") - state.last_match_date.floor("D")).days
    return float(np.clip(days, 1, 60))


class FeatureEngine:
    """Build pre-match features and retain state for future fixtures."""

    def __init__(self, initial_elo: float = DEFAULT_ELO) -> None:
        self.initial_elo = initial_elo
        self.states: dict[str, TeamState] = {}

    def copy(self) -> "FeatureEngine":
        return copy.deepcopy(self)

    def state_for(self, team: str) -> TeamState:
        name = canonical_team(team)
        if name not in self.states:
            self.states[name] = TeamState(elo=self.initial_elo)
        return self.states[name]

    def make_features(
        self,
        home_team: str,
        away_team: str,
        match_date: pd.Timestamp | str,
        *,
        tournament: str = "FIFA World Cup",
        stage: str = "GROUP_STAGE",
        neutral: bool = True,
    ) -> dict[str, float]:
        home = canonical_team(home_team)
        away = canonical_team(away_team)
        date = pd.to_datetime(match_date, utc=True)
        home_state = self.state_for(home)
        away_state = self.state_for(away)
        home_rest = _rest_days(home_state, date)
        away_rest = _rest_days(away_state, date)

        values = {
            "elo_home": home_state.elo,
            "elo_away": away_state.elo,
            "elo_diff": home_state.elo - away_state.elo,
            "home_form_5": _points(home_state.outcomes, 5),
            "away_form_5": _points(away_state.outcomes, 5),
            "home_form_10": _points(home_state.outcomes, 10),
            "away_form_10": _points(away_state.outcomes, 10),
            "home_goals_for_5": _mean(home_state.goals_for, 5),
            "away_goals_for_5": _mean(away_state.goals_for, 5),
            "home_goals_against_5": _mean(home_state.goals_against, 5),
            "away_goals_against_5": _mean(away_state.goals_against, 5),
            "home_win_rate_10": _rate(home_state.outcomes, "W"),
            "away_win_rate_10": _rate(away_state.outcomes, "W"),
            "home_draw_rate_10": _rate(home_state.outcomes, "D"),
            "away_draw_rate_10": _rate(away_state.outcomes, "D"),
            "home_loss_rate_10": _rate(home_state.outcomes, "L"),
            "away_loss_rate_10": _rate(away_state.outcomes, "L"),
            "home_rest_days": home_rest,
            "away_rest_days": away_rest,
            "neutral": float(neutral),
            "home_advantage": float(not neutral),
            "home_is_host": float(home in HOST_TEAMS),
            "away_is_host": float(away in HOST_TEAMS),
            "competition_importance": competition_importance(tournament, stage),
        }
        values.update(
            {
                "form_5_diff": values["home_form_5"] - values["away_form_5"],
                "form_10_diff": values["home_form_10"] - values["away_form_10"],
                "goals_for_5_diff": (
                    values["home_goals_for_5"] - values["away_goals_for_5"]
                ),
                "goals_against_5_diff": (
                    values["home_goals_against_5"]
                    - values["away_goals_against_5"]
                ),
                "rest_days_diff": home_rest - away_rest,
            }
        )
        return {column: float(values[column]) for column in FEATURE_COLUMNS}

    def _update(
        self,
        home_team: str,
        away_team: str,
        match_date: pd.Timestamp,
        home_score: int,
        away_score: int,
        *,
        tournament: str,
        stage: str,
        neutral: bool,
    ) -> None:
        home = self.state_for(home_team)
        away = self.state_for(away_team)
        result = regulation_result(home_score, away_score)

        home_result = "W" if result == "H" else "L" if result == "A" else "D"
        away_result = "W" if result == "A" else "L" if result == "H" else "D"
        home.outcomes.append(home_result)
        away.outcomes.append(away_result)
        home.goals_for.append(home_score)
        home.goals_against.append(away_score)
        away.goals_for.append(away_score)
        away.goals_against.append(home_score)

        advantage = 0.0 if neutral else HOME_ELO_ADVANTAGE
        expected_home = 1.0 / (1.0 + 10 ** (-(home.elo + advantage - away.elo) / 400))
        actual_home = 1.0 if result == "H" else 0.0 if result == "A" else 0.5
        margin = abs(home_score - away_score)
        margin_multiplier = 1.0 if margin <= 1 else 1.0 + math.log(margin)
        k = 8.0 * competition_importance(tournament, stage)
        change = k * margin_multiplier * (actual_home - expected_home)
        home.elo += change
        away.elo -= change
        home.last_match_date = match_date
        away.last_match_date = match_date
        home.matches += 1
        away.matches += 1

    def update_completed_matches(self, matches: pd.DataFrame) -> None:
        """Update team state from completed matches without creating features."""
        completed = matches.loc[
            matches["home_score"].notna() & matches["away_score"].notna()
        ].copy()
        completed["date"] = pd.to_datetime(completed["date"], utc=True)
        for row in completed.sort_values(["date", "home_team", "away_team"]).to_dict(
            "records"
        ):
            self._update(
                str(row["home_team"]),
                str(row["away_team"]),
                pd.to_datetime(row["date"], utc=True),
                int(row["home_score"]),
                int(row["away_score"]),
                tournament=str(row["tournament"]),
                stage=str(row["stage"]),
                neutral=bool(row["neutral"]),
            )

    def build_training_set(
        self, matches: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        completed = matches.loc[
            matches["home_score"].notna() & matches["away_score"].notna()
        ].copy()
        completed["date"] = pd.to_datetime(completed["date"], utc=True)
        completed = completed.sort_values(["date", "home_team", "away_team"])
        feature_rows: list[dict[str, float]] = []
        labels: list[str] = []
        audit_rows: list[dict[str, object]] = []

        # Matches on the same UTC day are all featurized before any result from
        # that day updates state. This is conservative and prevents subtle
        # same-day leakage when kickoff ordering is incomplete.
        for _, day_matches in completed.groupby(completed["date"].dt.floor("D")):
            pending_updates: list[dict[str, object]] = []
            for row in day_matches.to_dict("records"):
                home_state = self.state_for(row["home_team"])
                away_state = self.state_for(row["away_team"])
                feature_rows.append(
                    self.make_features(
                        row["home_team"],
                        row["away_team"],
                        row["date"],
                        tournament=row["tournament"],
                        stage=row["stage"],
                        neutral=bool(row["neutral"]),
                    )
                )
                labels.append(
                    regulation_result(int(row["home_score"]), int(row["away_score"]))
                )
                audit_rows.append(
                    {
                        "date": row["date"],
                        "home_team": row["home_team"],
                        "away_team": row["away_team"],
                        "home_history_cutoff": home_state.last_match_date,
                        "away_history_cutoff": away_state.last_match_date,
                    }
                )
                pending_updates.append(row)
            for row in pending_updates:
                self._update(
                    str(row["home_team"]),
                    str(row["away_team"]),
                    pd.to_datetime(row["date"], utc=True),
                    int(row["home_score"]),
                    int(row["away_score"]),
                    tournament=str(row["tournament"]),
                    stage=str(row["stage"]),
                    neutral=bool(row["neutral"]),
                )

        X = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)
        y = pd.Series(labels, name="result")
        audit = pd.DataFrame(audit_rows)
        return X, y, audit
