import numpy as np
import pandas as pd

from fifa_predict.players import (
    PlayerContextStore,
    adjust_probabilities,
    convert_fc25_ratings,
    parse_world_cup_rosters_html,
)


def _ratings(team: str, prefix: str, rating: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": f"{prefix}_{index}",
                "player_name": f"{team} Player {index}",
                "national_team": team,
                "overall_rating": rating - index // 6,
                "position": "MF",
                "rating_date": "2026-06-01T00:00:00Z",
                "source": "test",
            }
            for index in range(18)
        ]
    )


def test_player_quality_moves_decisive_probability_in_expected_direction() -> None:
    ratings = pd.concat(
        [_ratings("United States", "usa", 84), _ratings("Paraguay", "par", 76)],
        ignore_index=True,
    )
    store = PlayerContextStore(ratings=ratings)
    context = store.match_context(
        "United States",
        "Paraguay",
        "2026-06-13T01:00:00Z",
        as_of="2026-06-12T20:00:00Z",
    )
    baseline = np.array([0.40, 0.25, 0.35])
    adjusted = adjust_probabilities(baseline, context)

    assert adjusted[0] > baseline[0]
    assert adjusted[1] == baseline[1]
    np.testing.assert_allclose(adjusted.sum(), 1.0)


def test_late_injury_report_is_not_used() -> None:
    ratings = _ratings("United States", "usa", 82)
    availability = pd.DataFrame(
        [
            {
                "fixture_id": "1",
                "match_date": "2026-06-13T01:00:00Z",
                "team": "United States",
                "player_id": "usa_0",
                "status": "injured",
                "availability_probability": 0.0,
                "reported_at": "2026-06-13T02:00:00Z",
                "source": "test",
            }
        ]
    )
    store = PlayerContextStore(ratings=ratings, availability=availability)
    context = store.team_context(
        "United States",
        "2026-06-13T01:00:00Z",
        fixture_id="1",
        as_of="2026-06-14T00:00:00Z",
    )
    assert context.unavailable_quality == 0
    assert context.availability_rate == 1


def test_injury_reduces_availability_and_expected_lineup_quality() -> None:
    ratings = _ratings("United States", "usa", 90)
    availability = pd.DataFrame(
        [
            {
                "fixture_id": "1",
                "match_date": "2026-06-13T01:00:00Z",
                "team": "United States",
                "player_id": "usa_0",
                "status": "injured",
                "availability_probability": 0.0,
                "reported_at": "2026-06-12T18:00:00Z",
                "source": "test",
            }
        ]
    )
    store = PlayerContextStore(ratings=ratings, availability=availability)
    context = store.team_context(
        "United States",
        "2026-06-13T01:00:00Z",
        fixture_id="1",
        as_of="2026-06-12T20:00:00Z",
    )
    assert context.availability_rate < 1
    assert context.unavailable_quality > 0
    assert context.expected_starters == 11


def test_missing_opponent_data_leaves_probability_unchanged() -> None:
    store = PlayerContextStore(ratings=_ratings("United States", "usa", 82))
    context = store.match_context(
        "United States", "Paraguay", "2026-06-13T01:00:00Z"
    )
    baseline = np.array([0.4, 0.25, 0.35])
    np.testing.assert_allclose(adjust_probabilities(baseline, context), baseline)


def test_eleven_announced_starters_mark_lineup_confirmed() -> None:
    ratings = _ratings("United States", "usa", 82)
    lineups = pd.DataFrame(
        [
            {
                "fixture_id": "1",
                "match_date": "2026-06-13T01:00:00Z",
                "team": "United States",
                "player_id": f"usa_{index}",
                "lineup_status": "starter",
                "announced_at": "2026-06-12T23:45:00Z",
                "source": "test",
            }
            for index in range(11)
        ]
    )
    store = PlayerContextStore(ratings=ratings, lineups=lineups)
    context = store.team_context(
        "United States",
        "2026-06-13T01:00:00Z",
        fixture_id="1",
        as_of="2026-06-13T00:00:00Z",
    )
    assert context.lineup_confirmed
    assert context.expected_starters == 11


def test_fc25_converter_maps_nationality_and_quality() -> None:
    source = pd.DataFrame(
        [
            {
                "Name": "Test Player",
                "Nationality": "USA",
                "Club": "Test Club",
                "Overall": 81,
                "Position": "CM",
            }
        ]
    )
    converted = convert_fc25_ratings(source)
    assert converted.iloc[0]["national_team"] == "United States"
    assert converted.iloc[0]["overall_rating"] == 81
    assert len(converted.iloc[0]["player_id"]) == 16


def test_fc26_converter_uses_ea_id_and_common_name() -> None:
    source = pd.DataFrame(
        [
            {
                "id": 12345,
                "firstName": "Example",
                "lastName": "Player",
                "commonName": "Example",
                "nationality": "Korea Republic",
                "team": "Example FC",
                "overallRating": 79,
                "position": "RW",
            }
        ]
    )
    converted = convert_fc25_ratings(source)
    assert converted.iloc[0]["player_id"] == "ea_12345"
    assert converted.iloc[0]["player_name"] == "Example"
    assert converted.iloc[0]["national_team"] == "South Korea"


def test_registered_roster_is_authoritative_and_missing_rating_is_imputed() -> None:
    ratings = pd.DataFrame(
        [
            {
                "player_id": "ea_known",
                "player_name": "Known Player",
                "national_team": "South Africa",
                "overall_rating": 70,
                "position": "CB",
                "rating_date": "2026-03-18T00:00:00Z",
                "source": "test",
            },
            {
                "player_id": "ea_outsider",
                "player_name": "Unregistered Star",
                "national_team": "South Africa",
                "overall_rating": 90,
                "position": "ST",
                "rating_date": "2026-03-18T00:00:00Z",
                "source": "test",
            },
        ]
    )
    html = """
    <h3>South Africa</h3>
    <table>
      <tr><th>No.</th><th>Pos.</th><th>Player</th><th>Club</th></tr>
      <tr><td>1</td><td>1 GK</td><td>Missing Keeper</td><td>Club A</td></tr>
      <tr><td>2</td><td>2 DF</td><td>Known Player</td><td>Club B</td></tr>
    </table>
    """
    rosters = parse_world_cup_rosters_html(
        html,
        PlayerContextStore(ratings=ratings).ratings,
        retrieved_at="2026-06-13T00:00:00Z",
    )
    store = PlayerContextStore(ratings=ratings, rosters=rosters)
    squad = store.squad_for_match("South Africa", "2026-06-11T19:00:00Z")

    assert set(squad["player_name"]) == {"Missing Keeper", "Known Player"}
    assert "Unregistered Star" not in set(squad["player_name"])
    missing = squad.loc[squad["player_name"] == "Missing Keeper"].iloc[0]
    assert missing["rating_imputed"]
    assert missing["overall_rating"] == 70
    assert squad["roster_verified"].all()
