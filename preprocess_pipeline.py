from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _minmax_scale(values: np.ndarray, feature_min: np.ndarray, feature_max: np.ndarray) -> np.ndarray:
    denom = feature_max - feature_min
    denom[denom == 0] = 1.0
    return (values - feature_min) / denom


def _normalize_timestamp(timestamp: Optional[str]) -> Optional[str]:
    if timestamp is None:
        return None
    parsed = pd.to_datetime(timestamp, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid timestamp format: {timestamp}")
    return parsed.isoformat()


@dataclass
class PreprocessingConfig:
    feature_names: List[str]
    feature_min: np.ndarray
    feature_max: np.ndarray
    timesteps: int

    @classmethod
    def load(cls, path: Path | str) -> "PreprocessingConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Preprocessing config not found at {path}")

        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)

        feature_names = data.get("feature_names")
        feature_min = np.asarray(data.get("feature_min"), dtype=np.float32)
        feature_max = np.asarray(data.get("feature_max"), dtype=np.float32)
        timesteps = int(data.get("timesteps", 10))

        if (
            not feature_names
            or len(feature_names) != len(feature_min)
            or len(feature_names) != len(feature_max)
        ):
            raise ValueError("Preprocessing config is inconsistent: unequal feature lengths.")

        return cls(
            feature_names=list(feature_names),
            feature_min=feature_min,
            feature_max=feature_max,
            timesteps=timesteps,
        )


class LSTMOnlinePreprocessor:
    """
    Applies Min-Max scaling to incoming observations using saved training statistics.
    """

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self._feature_count = len(config.feature_names)

    @property
    def feature_names(self) -> List[str]:
        return self.config.feature_names

    @property
    def timesteps(self) -> int:
        return self.config.timesteps

    def _validate_vector(self, values: Sequence[float]) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape != (self._feature_count,):
            raise ValueError(f"Expected {self._feature_count} features, received {array.shape[0]}.")
        return array

    def scale(self, values: Sequence[float]) -> np.ndarray:
        vector = self._validate_vector(values)
        return _minmax_scale(vector, self.config.feature_min, self.config.feature_max)

    def scale_batch(self, batch: Sequence[Sequence[float]]) -> np.ndarray:
        if not batch:
            raise ValueError("Batch must contain at least one observation.")
        matrix = np.stack([self._validate_vector(row) for row in batch])
        df = pd.DataFrame(matrix)
        df = df.ffill().bfill()
        return _minmax_scale(df.values.astype(np.float32), self.config.feature_min, self.config.feature_max)


class WindowManager:
    """
    Maintains a rolling buffer of observations to build LSTM input windows.
    """

    def __init__(self, timesteps: int):
        if timesteps <= 0:
            raise ValueError("timesteps must be greater than zero.")
        self.timesteps = timesteps
        self._buffer: Deque[Tuple[np.ndarray, Optional[str]]] = deque(maxlen=timesteps)

    def reset(self) -> None:
        self._buffer.clear()

    def size(self) -> int:
        return len(self._buffer)

    def add(self, vector: np.ndarray, timestamp: Optional[str]) -> Tuple[Optional[np.ndarray], Optional[str]]:
        normalized_timestamp = _normalize_timestamp(timestamp) if timestamp else None
        self._buffer.append((vector, normalized_timestamp))

        if len(self._buffer) < self.timesteps:
            return None, None

        sequence = np.stack([entry[0] for entry in self._buffer], axis=0)
        last_timestamp = self._buffer[-1][1]
        return sequence, last_timestamp


__all__ = ["PreprocessingConfig", "LSTMOnlinePreprocessor", "WindowManager"]
