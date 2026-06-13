from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fifa_predict.features import FEATURE_COLUMNS, FeatureEngine

CLASSES = np.array(["H", "D", "A"])


def _align_probabilities(
    probabilities: np.ndarray, estimator_classes: np.ndarray
) -> np.ndarray:
    aligned = np.zeros((len(probabilities), len(CLASSES)), dtype=float)
    for target_index, label in enumerate(CLASSES):
        matches = np.where(estimator_classes == label)[0]
        if len(matches):
            aligned[:, target_index] = probabilities[:, matches[0]]
    totals = aligned.sum(axis=1, keepdims=True)
    return np.divide(
        aligned,
        totals,
        out=np.full_like(aligned, 1 / len(CLASSES)),
        where=totals > 0,
    )


class EloBaseline(BaseEstimator, ClassifierMixin):
    """Three-way Davidson-style baseline derived from the Elo difference."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EloBaseline":
        self.classes_ = CLASSES.copy()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        difference = (
            np.asarray(X["elo_diff"], dtype=float)
            + 60.0 * np.asarray(X["home_advantage"], dtype=float)
            + 35.0
            * (
                np.asarray(X["home_is_host"], dtype=float)
                - np.asarray(X["away_is_host"], dtype=float)
            )
        )
        home_decisive = 1.0 / (1.0 + 10 ** (-difference / 400.0))
        draw = np.clip(0.29 * np.exp(-np.abs(difference) / 550.0), 0.11, 0.30)
        home = (1.0 - draw) * home_decisive
        away = (1.0 - draw) * (1.0 - home_decisive)
        return np.column_stack([home, draw, away])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return CLASSES[np.argmax(self.predict_proba(X), axis=1)]


class ProbabilityCalibrator:
    """One-vs-rest sigmoid calibration fitted on time-safe OOF predictions."""

    def __init__(self) -> None:
        self.models: list[LogisticRegression | None] = []

    def fit(self, probabilities: np.ndarray, y: pd.Series) -> "ProbabilityCalibrator":
        self.models = []
        labels = np.asarray(y)
        for index, label in enumerate(CLASSES):
            target = (labels == label).astype(int)
            if len(np.unique(target)) < 2:
                self.models.append(None)
                continue
            feature = self._logit(probabilities[:, index]).reshape(-1, 1)
            model = LogisticRegression(C=1.0, solver="lbfgs")
            model.fit(feature, target)
            self.models.append(model)
        return self

    @staticmethod
    def _logit(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, 1e-6, 1 - 1e-6)
        return np.log(clipped / (1 - clipped))

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        calibrated = np.zeros_like(probabilities, dtype=float)
        for index, model in enumerate(self.models):
            if model is None:
                calibrated[:, index] = probabilities[:, index]
            else:
                feature = self._logit(probabilities[:, index]).reshape(-1, 1)
                calibrated[:, index] = model.predict_proba(feature)[:, 1]
        totals = calibrated.sum(axis=1, keepdims=True)
        return np.divide(
            calibrated,
            totals,
            out=np.full_like(calibrated, 1 / len(CLASSES)),
            where=totals > 0,
        )


def multiclass_brier(y: pd.Series, probabilities: np.ndarray) -> float:
    encoded = np.column_stack([(np.asarray(y) == label) for label in CLASSES])
    return float(np.mean(np.sum((probabilities - encoded) ** 2, axis=1)))


def expected_calibration_error(
    y: pd.Series, probabilities: np.ndarray, bins: int = 10
) -> float:
    labels = np.asarray(y)
    errors: list[float] = []
    edges = np.linspace(0, 1, bins + 1)
    for index, label in enumerate(CLASSES):
        observed = (labels == label).astype(float)
        for lower, upper in zip(edges[:-1], edges[1:]):
            include = (probabilities[:, index] >= lower) & (
                probabilities[:, index] < upper
                if upper < 1
                else probabilities[:, index] <= upper
            )
            if include.any():
                errors.append(
                    float(include.mean())
                    * abs(
                        float(probabilities[include, index].mean())
                        - float(observed[include].mean())
                    )
                )
    return float(sum(errors) / len(CLASSES))


def candidate_estimators(random_state: int = 42) -> dict[str, BaseEstimator]:
    return {
        "elo": EloBaseline(),
        "logistic": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.5,
                        max_iter=1500,
                        solver="lbfgs",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=180,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=random_state,
        ),
    }


def _metrics(y: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    class_indices = {label: index for index, label in enumerate(CLASSES)}
    true_indices = np.array([class_indices[label] for label in y], dtype=int)
    selected_probabilities = probabilities[np.arange(len(y)), true_indices]
    return {
        "log_loss": float(-np.mean(np.log(np.clip(selected_probabilities, 1e-15, 1)))),
        "brier_score": multiclass_brier(y, probabilities),
        "accuracy": float(
            accuracy_score(y, CLASSES[np.argmax(probabilities, axis=1)])
        ),
        "calibration_error": expected_calibration_error(y, probabilities),
    }


@dataclass
class ModelBundle:
    estimator: BaseEstimator
    calibrator: ProbabilityCalibrator
    feature_engine: FeatureEngine
    model_name: str
    model_version: str
    trained_through: str
    metrics: dict[str, Any]

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        raw = self.estimator.predict_proba(features[FEATURE_COLUMNS])
        aligned = _align_probabilities(raw, np.asarray(self.estimator.classes_))
        return self.calibrator.transform(aligned)


def train_and_select(
    X: pd.DataFrame,
    y: pd.Series,
    feature_engine: FeatureEngine,
    match_dates: pd.Series,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> ModelBundle:
    if len(X) < 100:
        raise ValueError("At least 100 completed matches are required for training")
    estimators = candidate_estimators(random_state)
    splitter = TimeSeriesSplit(n_splits=n_splits)
    candidate_oof = {
        name: np.full((len(X), len(CLASSES)), np.nan)
        for name in estimators
    }
    naive_oof = np.full((len(X), len(CLASSES)), np.nan)

    for train_indices, validation_indices in splitter.split(X):
        X_train = X.iloc[train_indices]
        y_train = y.iloc[train_indices]
        X_validation = X.iloc[validation_indices]
        counts = y_train.value_counts(normalize=True)
        naive_oof[validation_indices] = np.array(
            [counts.get(label, 0.0) for label in CLASSES]
        )
        for name, estimator in estimators.items():
            fitted = clone(estimator).fit(X_train, y_train)
            raw = fitted.predict_proba(X_validation)
            candidate_oof[name][validation_indices] = _align_probabilities(
                raw, np.asarray(fitted.classes_)
            )

    valid = ~np.isnan(naive_oof).any(axis=1)
    y_valid = y.loc[valid].reset_index(drop=True)
    metrics: dict[str, Any] = {
        "naive": _metrics(y_valid, naive_oof[valid]),
        "candidates": {},
        "validation_rows": int(valid.sum()),
    }
    for name, probabilities in candidate_oof.items():
        metrics["candidates"][name] = _metrics(y_valid, probabilities[valid])

    simple_best = min(
        ("elo", "logistic"),
        key=lambda name: metrics["candidates"][name]["log_loss"],
    )
    overall_best = min(
        estimators, key=lambda name: metrics["candidates"][name]["log_loss"]
    )
    # Complexity must buy a measurable log-loss improvement.
    if (
        overall_best == "hist_gradient_boosting"
        and metrics["candidates"][overall_best]["log_loss"]
        >= metrics["candidates"][simple_best]["log_loss"] - 0.001
    ):
        selected = simple_best
    else:
        selected = overall_best
    if (
        metrics["candidates"][selected]["log_loss"]
        >= metrics["naive"]["log_loss"]
    ):
        selected = simple_best
    if (
        metrics["candidates"][selected]["log_loss"]
        >= metrics["naive"]["log_loss"]
    ):
        raise RuntimeError(
            "No candidate model beat the naive class-frequency baseline."
        )

    selected_oof = candidate_oof[selected][valid]
    calibrator = ProbabilityCalibrator().fit(selected_oof, y_valid)
    calibrated_oof = calibrator.transform(selected_oof)
    metrics["selected"] = selected
    metrics["selected_raw"] = metrics["candidates"][selected]
    metrics["selected_calibrated"] = _metrics(y_valid, calibrated_oof)
    metrics["beats_naive"] = (
        metrics["selected_raw"]["log_loss"] < metrics["naive"]["log_loss"]
    )

    fitted_estimator = clone(estimators[selected]).fit(X, y)
    trained_through = pd.to_datetime(match_dates, utc=True).max().isoformat()
    model_version = (
        f"{selected}-through-{pd.Timestamp(trained_through).strftime('%Y%m%d')}"
    )
    return ModelBundle(
        estimator=fitted_estimator,
        calibrator=calibrator,
        feature_engine=feature_engine.copy(),
        model_name=selected,
        model_version=model_version,
        trained_through=trained_through,
        metrics=metrics,
    )
