import pandas as pd
import pytest

from weather_calibration.features import FEATURE_COLUMNS, TARGET_COLUMN, prepare_training_frame
from weather_calibration.pipeline import CSVDataSource, CalibrationPipeline
from weather_calibration.synthetic import make_synthetic_weather_frame


def test_prepare_training_frame_builds_target_and_sorts_dates():
    raw = make_synthetic_weather_frame(days=140, seed=7).sample(frac=1, random_state=1)

    frame = prepare_training_frame(raw)

    assert frame["date"].is_monotonic_increasing
    expected_target = frame["observed_target"] - frame["source_forecast"]
    pd.testing.assert_series_equal(
        frame[TARGET_COLUMN],
        expected_target,
        check_names=False,
    )


def test_prepare_training_frame_fails_on_missing_feature():
    raw = make_synthetic_weather_frame(days=140, seed=7).drop(columns=[FEATURE_COLUMNS[0]])

    with pytest.raises(ValueError, match=FEATURE_COLUMNS[0]):
        prepare_training_frame(raw)


def test_csv_pipeline_loads_and_transforms(tmp_path):
    path = tmp_path / "synthetic.csv"
    make_synthetic_weather_frame(days=140, seed=9).to_csv(path, index=False)

    pipeline = CalibrationPipeline(CSVDataSource(path))
    frame = pipeline.run()

    assert TARGET_COLUMN in frame.columns
    assert len(frame) == 140
