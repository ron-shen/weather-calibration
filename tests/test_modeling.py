from weather_calibration.features import prepare_training_frame
from weather_calibration.modeling import train_calibrator
from weather_calibration.synthetic import make_synthetic_weather_frame
from xgboost import XGBRegressor


def test_train_calibrator_beats_uncalibrated_baseline():
    raw = make_synthetic_weather_frame(days=420, seed=99)
    frame = prepare_training_frame(raw)

    result = train_calibrator(frame, test_size=0.2)

    assert result.report.train_rows == 336
    assert result.report.test_rows == 84
    assert isinstance(result.estimator, XGBRegressor)
    assert "max_depth" in result.report.best_params
    assert result.report.model_rmse < result.report.baseline_rmse
    assert set(result.predictions.columns) >= {
        "date",
        "source_forecast",
        "observed_target",
        "forecast_error",
        "predicted_error",
        "calibrated_forecast",
    }
