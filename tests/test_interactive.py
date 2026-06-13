import pandas as pd
import pytest

from fifa_predict.interactive import find_fixture, resolve_team, review_lineup
from fifa_predict.players import PlayerContextStore


def _squad() -> pd.DataFrame:
    positions = [
        "GK",
        "GK",
        "CB",
        "CB",
        "CB",
        "LB",
        "RB",
        "LWB",
        "CDM",
        "CM",
        "CM",
        "CAM",
        "LM",
        "ST",
        "RW",
        "LW",
        "CF",
        "ST",
    ]
    return pd.DataFrame(
        [
            {
                "player_id": f"player_{index}",
                "player_name": f"Player {index}",
                "position": position,
                "overall_rating": 90 - index,
                "availability_probability": 1.0,
                "availability_reported": False,
                "lineup_status": pd.NA,
            }
            for index, position in enumerate(positions)
        ]
    )


def test_resolve_team_accepts_alias_and_typo() -> None:
    teams = ["United States", "Paraguay", "South Korea"]
    assert resolve_team("USA", teams) == "United States"
    assert resolve_team("Paraguy", teams) == "Paraguay"


def test_find_fixture_accepts_teams_in_either_order() -> None:
    fixtures = pd.DataFrame(
        [
            {
                "home_team": "United States",
                "away_team": "Paraguay",
                "date": pd.Timestamp("2026-06-13T01:00:00Z"),
            }
        ]
    )
    fixture = find_fixture(fixtures, "Paraguay", "USA")
    assert fixture["home_team"] == "United States"


def test_proposed_lineup_has_eleven_and_a_goalkeeper() -> None:
    squad = _squad()
    lineup = PlayerContextStore.propose_lineup(squad)
    selected = squad.loc[squad["player_id"].isin(lineup)]
    assert len(lineup) == 11
    assert selected["position"].eq("GK").sum() == 1


def test_known_lineup_takes_priority_over_balanced_proposal() -> None:
    squad = _squad()
    expected = squad.tail(11)["player_id"].tolist()
    squad.loc[squad["player_id"].isin(expected), "lineup_status"] = "starter"
    assert PlayerContextStore.propose_lineup(squad) == expected


def test_review_lineup_accepts_proposal() -> None:
    answers = iter([""])
    output: list[str] = []
    lineup = review_lineup(
        "United States",
        _squad(),
        input_fn=lambda _: next(answers),
        output=output.append,
    )
    assert len(lineup) == 11
    assert any("United States squad" in line for line in output)


def test_review_lineup_rejects_unavailable_selection_then_accepts() -> None:
    squad = _squad()
    squad.loc[0, "availability_probability"] = 0.0
    answers = iter(
        [
            "n",
            "1,2,3,4,5,6,7,8,9,10,11",
            "n",
            "2,3,4,5,6,7,8,9,10,11,12",
        ]
    )
    output: list[str] = []
    lineup = review_lineup(
        "United States",
        squad,
        input_fn=lambda _: next(answers),
        output=output.append,
    )
    assert len(lineup) == 11
    assert "player_0" not in lineup
    assert any("unavailable" in line.casefold() for line in output)


def test_unknown_team_is_rejected() -> None:
    with pytest.raises(ValueError):
        resolve_team("Atlantis", ["United States", "Paraguay"])
