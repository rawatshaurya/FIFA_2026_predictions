from __future__ import annotations

import difflib
from collections.abc import Callable

import joblib
import pandas as pd

from fifa_predict import config
from fifa_predict.data import load_world_cup_matches
from fifa_predict.players import PlayerContextStore
from fifa_predict.predictor import MatchPredictor
from fifa_predict.teams import canonical_team

Input = Callable[[str], str]
Output = Callable[[str], None]


def resolve_team(value: str, teams: list[str]) -> str:
    canonical = canonical_team(value)
    exact = {team.casefold(): team for team in teams}
    if canonical.casefold() in exact:
        return exact[canonical.casefold()]
    matches = difflib.get_close_matches(
        canonical.casefold(), list(exact), n=1, cutoff=0.55
    )
    if not matches:
        raise ValueError(f"Team not found: {value}")
    return exact[matches[0]]


def find_fixture(
    fixtures: pd.DataFrame, first_team: str, second_team: str
) -> pd.Series:
    teams = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))
    first = resolve_team(first_team, teams)
    second = resolve_team(second_team, teams)
    if first == second:
        raise ValueError("Please choose two different teams")
    matching = fixtures.loc[
        (
            (fixtures["home_team"] == first)
            & (fixtures["away_team"] == second)
        )
        | (
            (fixtures["home_team"] == second)
            & (fixtures["away_team"] == first)
        )
    ].sort_values("date")
    if matching.empty:
        raise ValueError(f"No World Cup fixture found for {first} vs {second}")
    return matching.iloc[0]


def _show_squad(
    team: str,
    squad: pd.DataFrame,
    proposed: list[str],
    output: Output,
) -> None:
    output(f"\n{team} squad")
    verified = (
        bool(squad["roster_verified"].all())
        if "roster_verified" in squad
        else False
    )
    output(
        "Registered World Cup roster"
        if verified
        else "WARNING: registered roster unavailable; showing rating candidates"
    )
    output(" #  XI  Player                         Pos  OVR   Rating       Availability")
    for index, row in squad.reset_index(drop=True).iterrows():
        selected = "*" if str(row["player_id"]) in proposed else " "
        availability = float(row["availability_probability"])
        reported = bool(row.get("availability_reported", False))
        status = (
            "OUT"
            if availability == 0
            else "DOUBTFUL"
            if availability < 1
            else "Available"
            if reported
            else "No report"
        )
        output(
            f"{index + 1:>2}   {selected}  "
            f"{str(row['player_name'])[:28]:<28} "
            f"{str(row['position'])[:4]:<4} "
            f"{int(row['overall_rating']):>3}   "
            f"{'imputed' if bool(row.get('rating_imputed', False)) else 'FC 26':<11} "
            f"{status}"
        )


def review_lineup(
    team: str,
    squad: pd.DataFrame,
    *,
    input_fn: Input = input,
    output: Output = print,
) -> list[str]:
    proposed = PlayerContextStore.propose_lineup(squad)
    while True:
        _show_squad(team, squad, proposed, output)
        answer = input_fn(
            f"\nUse the proposed {team} XI? [Y/n]: "
        ).strip().casefold()
        if answer in {"", "y", "yes"}:
            return proposed
        raw = input_fn(
            "Enter exactly 11 squad numbers separated by commas: "
        ).strip()
        try:
            numbers = [int(value.strip()) for value in raw.split(",")]
        except ValueError:
            output("Use numbers only, for example: 1,2,3,4,5,6,7,8,9,10,11")
            continue
        if len(numbers) != 11 or len(set(numbers)) != 11:
            output("Please choose exactly 11 unique squad numbers.")
            continue
        if min(numbers) < 1 or max(numbers) > len(squad):
            output(f"Squad numbers must be between 1 and {len(squad)}.")
            continue
        chosen = squad.iloc[[number - 1 for number in numbers]]
        if (chosen["availability_probability"] <= 0).any():
            output("An unavailable player cannot be selected.")
            continue
        return chosen["player_id"].astype(str).tolist()


def run_interactive(
    *,
    offline: bool = False,
    input_fn: Input = input,
    output: Output = print,
) -> None:
    fixtures = load_world_cup_matches(refresh=not offline, offline=offline)
    store = PlayerContextStore.from_csv()
    if not store.available:
        raise RuntimeError(
            "Player ratings are missing. Run: uv run fifa-predict players"
        )
    if not store.rosters_available:
        raise RuntimeError(
            "Registered World Cup rosters are missing. "
            "Run: uv run fifa-predict players"
        )
    if not config.MODEL_ARTIFACT.exists():
        raise RuntimeError(
            "The trained model is missing. Run: uv run fifa-predict train --offline"
        )

    output("\nFIFA 2026 Match Predictor")
    output("Type the two teams, review both starting elevens, then predict.\n")
    while True:
        first = input_fn("First team: ").strip()
        second = input_fn("Second team: ").strip()
        try:
            fixture = find_fixture(fixtures, first, second)
            break
        except ValueError as error:
            output(f"\n{error}\nPlease try again.\n")
    fixture_id = (
        None if pd.isna(fixture["fixture_id"]) else str(fixture["fixture_id"])
    )
    kickoff = pd.to_datetime(fixture["date"], utc=True)
    output(
        f"\nFixture: {fixture['home_team']} vs {fixture['away_team']}"
        f"\nKickoff: {kickoff.strftime('%Y-%m-%d %H:%M UTC')}"
        f"\nStage: {fixture['stage']}"
    )

    home_squad = store.squad_for_match(
        str(fixture["home_team"]), kickoff, fixture_id=fixture_id
    )
    away_squad = store.squad_for_match(
        str(fixture["away_team"]), kickoff, fixture_id=fixture_id
    )
    if len(home_squad) < 11 or len(away_squad) < 11:
        raise RuntimeError(
            "One of these teams has fewer than 11 rated players. "
            "Add ratings before using the interactive predictor."
        )
    home_xi = review_lineup(
        str(fixture["home_team"]),
        home_squad,
        input_fn=input_fn,
        output=output,
    )
    away_xi = review_lineup(
        str(fixture["away_team"]),
        away_squad,
        input_fn=input_fn,
        output=output,
    )
    confirmed_store = store.with_confirmed_lineups(
        str(fixture["home_team"]),
        str(fixture["away_team"]),
        kickoff,
        home_xi,
        away_xi,
        fixture_id=fixture_id,
    )
    predictor = MatchPredictor(
        joblib.load(config.MODEL_ARTIFACT),
        player_context=confirmed_store,
    )
    prior_completed = fixtures.loc[
        fixtures["home_score"].notna()
        & fixtures["away_score"].notna()
        & (fixtures["date"].dt.floor("D") < kickoff.floor("D"))
    ]
    predictor.feature_engine.update_completed_matches(prior_completed)
    prediction = predictor.predict_match(
        str(fixture["home_team"]),
        str(fixture["away_team"]),
        kickoff,
        str(fixture["stage"]),
        bool(fixture["neutral"]),
        fixture_id=fixture_id,
        source="interactive",
    )

    output("\nPrediction")
    output(
        f"{prediction.home_team} win: "
        f"{prediction.home_win_probability:.1%}"
    )
    output(f"Draw: {prediction.draw_probability:.1%}")
    output(
        f"{prediction.away_team} win: "
        f"{prediction.away_win_probability:.1%}"
    )
    output(f"Most likely: {prediction.predicted_result.replace('_', ' ').title()}")
    output(
        "XI quality: "
        f"{prediction.home_expected_xi_quality:.1f} vs "
        f"{prediction.away_expected_xi_quality:.1f}"
    )
