import pytest

from fifa_predict.teams import canonical_team


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("USA", "United States"),
        ("United States of America", "United States"),
        ("Korea Republic", "South Korea"),
        ("Côte d'Ivoire", "Ivory Coast"),
        ("Türkiye", "Turkey"),
        ("Bosnia & Herzegovina", "Bosnia and Herzegovina"),
    ],
)
def test_team_aliases(source: str, expected: str) -> None:
    assert canonical_team(source) == expected


def test_empty_team_rejected() -> None:
    with pytest.raises(ValueError):
        canonical_team("  ")
