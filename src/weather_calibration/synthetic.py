from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_synthetic_weather_frame(
    *,
    start: str = "2022-01-01",
    days: int = 900,
    seed: int = 42,
    n_features: int = 8,
) -> pd.DataFrame:
    """Create a deterministic synthetic calibration dataset.

    Feature columns are intentionally anonymous (`x_00`, `x_01`, ...). The
    values are generated from latent signals and noise, not from real stations,
    market behavior, private model outputs, or production feature definitions.
    """
    if days < 120:
        raise ValueError("days must be at least 120 to support time-series validation.")
    if n_features < 5:
        raise ValueError("n_features must be at least 5.")

    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=days, freq="D")
    t = np.arange(days)

    annual = np.sin(2 * np.pi * t / 365.25)
    shoulder = np.cos(2 * np.pi * t / 182.625)
    short_cycle = np.sin(2 * np.pi * t / 29.0)

    x = rng.normal(0, 1, size=(days, n_features))
    x[:, 0] = 0.75 * annual + rng.normal(0, 0.45, days)
    x[:, 1] = 0.60 * shoulder + rng.normal(0, 0.50, days)
    x[:, 2] = 0.55 * short_cycle + rng.normal(0, 0.65, days)
    x[:, 3] = np.tanh(x[:, 0] - x[:, 1]) + rng.normal(0, 0.35, days)
    x[:, 4] = rng.normal(0, 1, days)

    forecast_error = (
        1.20 * x[:, 0]
        - 0.85 * x[:, 1]
        + 0.45 * x[:, 2]
        + 0.35 * x[:, 3]
        - 0.20 * x[:, 4]
        + rng.normal(0, 1.15 + 0.20 * np.abs(annual), days)
    )

    latent_target = 62 + 18 * annual + 4 * shoulder + rng.normal(0, 1.25, days)
    source_forecast = latent_target - forecast_error

    frame = pd.DataFrame(
        {
            "date": dates,
            "source_forecast": source_forecast,
            "observed_target": latent_target,
        }
    )
    for idx in range(n_features):
        frame[f"x_{idx:02d}"] = x[:, idx]

    return frame.round(4)


def write_synthetic_weather_csv(
    path: str | Path,
    *,
    start: str = "2022-01-01",
    days: int = 900,
    seed: int = 42,
    n_features: int = 8,
) -> Path:
    """Write a synthetic dataset and return the resolved path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = make_synthetic_weather_frame(
        start=start,
        days=days,
        seed=seed,
        n_features=n_features,
    )
    frame.to_csv(output_path, index=False)
    return output_path.resolve()
