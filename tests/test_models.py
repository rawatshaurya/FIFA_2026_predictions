import numpy as np
import pandas as pd

from fifa_predict.features import FEATURE_COLUMNS, FeatureEngine
from fifa_predict.models import train_and_select
from fifa_predict.predictor import MatchPredictor


def _synthetic_training_set(rows: int = 360):
    rng = np.random.default_rng(42)
    difference = rng.normal(0, 260, rows)
    draw_threshold = rng.random(rows)
    home_chance = 1 / (1 + 10 ** (-difference / 400))
    labels = np.where(
        draw_threshold < 0.2,
        "D",
        np.where(rng.random(rows) < home_chance, "H", "A"),
    )
    X = pd.DataFrame(0.0, index=range(rows), columns=FEATURE_COLUMNS)
    X["elo_home"] = 1500 + difference / 2
    X["elo_away"] = 1500 - difference / 2
    X["elo_diff"] = difference
    X["neutral"] = 1.0
    X["home_form_5"] = np.clip(0.5 + difference / 1200, 0, 1)
    X["away_form_5"] = np.clip(0.5 - difference / 1200, 0, 1)
    X["form_5_diff"] = X["home_form_5"] - X["away_form_5"]
    X["competition_importance"] = 2.5
    y = pd.Series(labels)
    dates = pd.Series(pd.date_range("2020-01-01", periods=rows, freq="D", tz="UTC"))
    return X, y, dates


def test_selected_model_beats_naive_and_probabilities_sum_to_one() -> None:
    X, y, dates = _synthetic_training_set()
    bundle = train_and_select(X, y, FeatureEngine(), dates, n_splits=4)
    assert bundle.metrics["beats_naive"]
    probabilities = bundle.predict_proba(X.tail(10))
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_knockout_has_eventual_winner_but_group_does_not() -> None:
    X, y, dates = _synthetic_training_set()
    bundle = train_and_select(X, y, FeatureEngine(), dates, n_splits=4)
    predictor = MatchPredictor(bundle, as_of="2026-06-13T12:00:00Z")
    group = predictor.predict_match(
        "Mexico", "Canada", "2026-06-20", "GROUP_A", True
    )
    knockout = predictor.predict_match(
        "Mexico", "Canada", "2026-07-01", "ROUND_OF_32", True
    )
    assert group.eventual_home_probability is None
    assert group.eventual_away_probability is None
    assert knockout.eventual_home_probability is not None
    assert knockout.eventual_away_probability is not None
    assert abs(
        knockout.eventual_home_probability
        + knockout.eventual_away_probability
        - 1
    ) < 1e-9


def test_tournament_timeline_includes_actual_results_without_same_day_leakage() -> None:
    X, y, dates = _synthetic_training_set()
    bundle = train_and_select(X, y, FeatureEngine(), dates, n_splits=4)
    fixtures = pd.DataFrame(
        [
            {
                "fixture_id": "1",
                "date": "2026-06-11T18:00:00Z",
                "home_team": "Alpha",
                "away_team": "Beta",
                "home_score": 3,
                "away_score": 0,
                "tournament": "FIFA World Cup",
                "stage": "GROUP_A",
                "neutral": True,
                "status": "FINISHED",
                "source": "test",
            },
            {
                "fixture_id": "2",
                "date": "2026-06-11T22:00:00Z",
                "home_team": "Alpha",
                "away_team": "Gamma",
                "home_score": 0,
                "away_score": 1,
                "tournament": "FIFA World Cup",
                "stage": "GROUP_A",
                "neutral": True,
                "status": "FINISHED",
                "source": "test",
            },
            {
                "fixture_id": "3",
                "date": "2026-06-12T18:00:00Z",
                "home_team": "Alpha",
                "away_team": "Delta",
                "home_score": pd.NA,
                "away_score": pd.NA,
                "tournament": "FIFA World Cup",
                "stage": "GROUP_A",
                "neutral": True,
                "status": "SCHEDULED",
                "source": "test",
            },
        ]
    )
    predictor = MatchPredictor(bundle, as_of="2026-06-13T12:00:00Z")
    records = predictor.predict_tournament_timeline(fixtures)

    assert records[0].actual_result == "home_win"
    assert records[0].prediction_correct in {True, False}
    assert records[1].actual_result == "away_win"
    assert records[2].actual_result is None
    assert records[2].prediction_correct is None
    # Both same-day Alpha fixtures were predicted from the identical pre-day
    # state; the first result was not allowed to alter the second prediction.
    assert records[0].home_win_probability == records[1].home_win_probability
