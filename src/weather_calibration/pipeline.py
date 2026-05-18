from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from weather_calibration.features import FEATURE_COLUMNS, prepare_training_frame


class DataSource(Protocol):
    def extract(self) -> pd.DataFrame: ...


@dataclass(frozen=True)
class CSVDataSource:
    path: str | Path

    def extract(self) -> pd.DataFrame:
        return pd.read_csv(self.path)


@dataclass(frozen=True)
class CalibrationPipeline:
    source: DataSource
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS

    def run(self) -> pd.DataFrame:
        raw = self.source.extract()
        return prepare_training_frame(raw, feature_columns=self.feature_columns)
