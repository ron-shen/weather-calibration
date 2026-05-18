import pandas as pd
import pytest

from weather_calibration.synthetic import make_synthetic_weather_frame


def test_synthetic_data_is_deterministic_and_anonymous():
    first = make_synthetic_weather_frame(days=180, seed=123)
    second = make_synthetic_weather_frame(days=180, seed=123)

    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns[:3]) == ["date", "source_forecast", "observed_target"]
    assert [column for column in first.columns if column.startswith("x_")] == [
        "x_00",
        "x_01",
        "x_02",
        "x_03",
        "x_04",
        "x_05",
        "x_06",
        "x_07",
    ]


def test_synthetic_data_rejects_too_few_rows():
    with pytest.raises(ValueError, match="days must be at least 120"):
        make_synthetic_weather_frame(days=60)
