import pandas as pd

from fifa_predict.features import DEFAULT_ELO, FeatureEngine


def _matches() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-01-01T12:00:00Z",
                "home_team": "Alpha",
                "away_team": "Beta",
                "home_score": 2,
                "away_score": 0,
                "tournament": "Friendly",
                "stage": "Friendly",
                "neutral": True,
            },
            {
                "date": "2026-01-01T18:00:00Z",
                "home_team": "Alpha",
                "away_team": "Gamma",
                "home_score": 0,
                "away_score": 1,
                "tournament": "Friendly",
                "stage": "Friendly",
                "neutral": True,
            },
            {
                "date": "2026-01-05T12:00:00Z",
                "home_team": "Alpha",
                "away_team": "Delta",
                "home_score": 1,
                "away_score": 1,
                "tournament": "Friendly",
                "stage": "Friendly",
                "neutral": True,
            },
        ]
    )


def test_same_day_results_do_not_enter_features() -> None:
    engine = FeatureEngine()
    X, _, audit = engine.build_training_set(_matches())
    assert X.iloc[0]["elo_home"] == DEFAULT_ELO
    assert X.iloc[1]["elo_home"] == DEFAULT_ELO
    known = audit["home_history_cutoff"].notna()
    assert (
        audit.loc[known, "home_history_cutoff"].dt.floor("D")
        < audit.loc[known, "date"].dt.floor("D")
    ).all()


def test_unseen_team_gets_documented_defaults() -> None:
    engine = FeatureEngine()
    features = engine.make_features(
        "Never Played", "Also New", "2026-06-20", neutral=True
    )
    assert features["elo_home"] == DEFAULT_ELO
    assert features["home_form_5"] == 0.5
    assert features["home_rest_days"] == 30
