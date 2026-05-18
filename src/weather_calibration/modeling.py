from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from weather_calibration.features import FEATURE_COLUMNS, TARGET_COLUMN, split_time_ordered


@dataclass(frozen=True)
class ModelReport:
    train_rows: int
    test_rows: int
    baseline_mae: float
    model_mae: float
    baseline_rmse: float
    model_rmse: float
    best_params: dict[str, Any]


@dataclass(frozen=True)
class TrainingResult:
    estimator: XGBRegressor
    report: ModelReport
    predictions: pd.DataFrame


def train_calibrator(
    frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    test_size: float | int = 0.2,
) -> TrainingResult:
    """Train an XGBoost forecast-error calibrator with chronological validation."""
    train_frame, test_frame = split_time_ordered(frame, test_size=test_size)

    X_train = train_frame.loc[:, feature_columns]
    y_train = train_frame[TARGET_COLUMN]
    X_test = test_frame.loc[:, feature_columns]
    y_test = test_frame[TARGET_COLUMN]

    estimator = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=7,
        n_jobs=1,
    )
    cv_test_size = max(12, min(60, len(train_frame) // 8))
    search = GridSearchCV(
        estimator=estimator,
        param_grid={
            "n_estimators": [80, 140],
            "max_depth": [2, 3],
            "learning_rate": [0.03, 0.08],
            "subsample": [0.85],
            "colsample_bytree": [0.85],
            "reg_lambda": [1.0, 5.0],
        },
        cv=TimeSeriesSplit(n_splits=5, test_size=cv_test_size),
        scoring="neg_root_mean_squared_error",
        refit=True,
    )
    search.fit(X_train, y_train)

    model_pred = search.predict(X_test)
    baseline_pred = np.zeros_like(y_test, dtype=float)

    report = ModelReport(
        train_rows=len(train_frame),
        test_rows=len(test_frame),
        baseline_mae=round(float(mean_absolute_error(y_test, baseline_pred)), 4),
        model_mae=round(float(mean_absolute_error(y_test, model_pred)), 4),
        baseline_rmse=round(float(np.sqrt(mean_squared_error(y_test, baseline_pred))), 4),
        model_rmse=round(float(np.sqrt(mean_squared_error(y_test, model_pred))), 4),
        best_params=dict(search.best_params_),
    )
    predictions = test_frame[["date", "source_forecast", "observed_target", TARGET_COLUMN]].copy()
    predictions["predicted_error"] = model_pred
    predictions["calibrated_forecast"] = predictions["source_forecast"] + model_pred

    return TrainingResult(
        estimator=search.best_estimator_,
        report=report,
        predictions=predictions.reset_index(drop=True),
    )


def save_training_outputs(
    result: TrainingResult,
    *,
    report_path: str | Path,
    model_path: str | Path | None = None,
) -> None:
    report_output = Path(report_path)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(asdict(result.report), indent=2), encoding="utf-8")

    if model_path is not None:
        model_output = Path(model_path)
        model_output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(result.estimator, model_output)
