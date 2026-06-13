"""FIFA World Cup match prediction toolkit."""

from typing import Any

__all__ = ["MatchPredictor", "PredictionRecord", "predict_match"]
__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    if name == "MatchPredictor":
        from fifa_predict.predictor import MatchPredictor

        return MatchPredictor
    if name == "PredictionRecord":
        from fifa_predict.schemas import PredictionRecord

        return PredictionRecord
    if name == "predict_match":
        from fifa_predict.pipeline import predict_match

        return predict_match
    raise AttributeError(name)
