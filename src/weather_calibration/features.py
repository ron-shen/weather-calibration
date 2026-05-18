from __future__ import annotations

import pandas as pd


FEATURE_COLUMNS = tuple(f"x_{idx:02d}" for idx in range(8))
DATE_COLUMN = "date"
FORECAST_COLUMN = "source_forecast"
OBSERVED_COLUMN = "observed_target"
TARGET_COLUMN = "forecast_error"


def prepare_training_frame(
    raw: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Convert raw synthetic rows into model-ready rows.

    The public project deliberately keeps feature names anonymous. This function
    only demonstrates validation, target construction, date ordering, and model
    input selection.
    """
    required = {DATE_COLUMN, FORECAST_COLUMN, OBSERVED_COLUMN, *feature_columns}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Input frame is missing required columns: {missing}")

    frame = raw.copy()
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], errors="raise")
    frame[TARGET_COLUMN] = frame[OBSERVED_COLUMN] - frame[FORECAST_COLUMN]
    frame = frame.sort_values(DATE_COLUMN).reset_index(drop=True)

    columns = [DATE_COLUMN, FORECAST_COLUMN, OBSERVED_COLUMN, *feature_columns, TARGET_COLUMN]
    return frame[columns]


def split_time_ordered(
    frame: pd.DataFrame,
    *,
    test_size: float | int = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a frame chronologically."""
    if isinstance(test_size, float):
        if not 0 < test_size < 1:
            raise ValueError("Float test_size must be between 0 and 1.")
        test_rows = max(1, int(round(len(frame) * test_size)))
    else:
        test_rows = int(test_size)

    if test_rows <= 0 or test_rows >= len(frame):
        raise ValueError("test_size must leave at least one train and one test row.")

    split_at = len(frame) - test_rows
    return frame.iloc[:split_at].copy(), frame.iloc[split_at:].copy()
