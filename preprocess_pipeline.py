from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler


@dataclass
class LSTMPreprocessor:
    """
    Preprocessing pipeline used to prepare SECOM sensor data for the LSTM model.
    The pipeline mirrors the training-time steps:
        1. Forward/backward fill missing values.
        2. Min-Max scaling.
        3. Variance threshold feature selection.
        4. Removal of highly correlated features (>|0.95|).
        5. Construction of sliding windows with a fixed number of timesteps.
    """

    scaler: MinMaxScaler
    variance_selector: VarianceThreshold
    selected_feature_names: List[str]
    correlated_drop_columns: List[str]
    final_feature_names: List[str]
    base_feature_names: List[str]
    timesteps: int

    @classmethod
    def fit_from_csv(cls, csv_path: Path | str, timesteps: int = 10) -> "LSTMPreprocessor":
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Training dataset not found at {csv_path}")

        data = pd.read_csv(csv_path)

        if "Time" in data.columns:
            time_index = pd.to_datetime(data["Time"])
            data = data.drop(columns=["Time"])
        else:
            time_index = pd.RangeIndex(len(data))

        if "Pass/Fail" in data.columns:
            data = data.drop(columns=["Pass/Fail"])

        base_feature_names = data.columns.tolist()
        if not base_feature_names:
            raise ValueError("No feature columns found after removing 'Time' and 'Pass/Fail'.")

        data = data.ffill().bfill()

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data)
        scaled_df = pd.DataFrame(scaled, index=time_index, columns=base_feature_names)

        variance_selector = VarianceThreshold(threshold=0.02)
        scaled_variance = variance_selector.fit_transform(scaled_df)
        selected_feature_indices = variance_selector.get_support(indices=True)
        selected_feature_names = [base_feature_names[i] for i in selected_feature_indices]
        variance_df = pd.DataFrame(scaled_variance, index=time_index, columns=selected_feature_names)

        correlation_matrix = variance_df.corr(method="pearson")
        upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        correlated_drop_columns = [
            column for column in upper_tri.columns if upper_tri[column].abs().max() > 0.95
        ]

        reduced_df = variance_df.drop(columns=correlated_drop_columns, errors="ignore")
        final_feature_names = reduced_df.columns.tolist()
        if not final_feature_names:
            raise ValueError("Final feature list is empty after correlation filtering.")

        return cls(
            scaler=scaler,
            variance_selector=variance_selector,
            selected_feature_names=selected_feature_names,
            correlated_drop_columns=correlated_drop_columns,
            final_feature_names=final_feature_names,
            base_feature_names=base_feature_names,
            timesteps=timesteps,
        )

    @property
    def base_feature_count(self) -> int:
        return len(self.base_feature_names)

    @property
    def final_feature_count(self) -> int:
        return len(self.final_feature_names)

    def transform(
        self,
        instances: Sequence[Sequence[float]],
        timestamps: Optional[Iterable[str]] = None,
    ) -> Tuple[np.ndarray, List[int], List[Optional[pd.Timestamp]]]:
        if not instances:
            raise ValueError("`instances` must contain at least one observation.")

        array = np.asarray(instances, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError("`instances` must be a 2D array [n_samples, n_features].")

        if array.shape[1] != self.base_feature_count:
            raise ValueError(
                f"Each observation must have {self.base_feature_count} features "
                f"(received {array.shape[1]})."
            )

        if timestamps is not None:
            timestamps = list(timestamps)
            if len(timestamps) != len(array):
                raise ValueError("Length of `timestamps` must match the number of instances.")
            index = pd.to_datetime(timestamps)
        else:
            index = pd.RangeIndex(len(array))

        df = pd.DataFrame(array, index=index, columns=self.base_feature_names).sort_index()
        df = df.ffill().bfill()

        scaled = self.scaler.transform(df)
        scaled_df = pd.DataFrame(scaled, index=df.index, columns=self.base_feature_names)

        variance_array = self.variance_selector.transform(scaled_df)
        variance_df = pd.DataFrame(variance_array, index=df.index, columns=self.selected_feature_names)

        reduced_df = variance_df.drop(columns=self.correlated_drop_columns, errors="ignore")
        reduced_df = reduced_df[self.final_feature_names]

        if len(reduced_df) < self.timesteps:
            raise ValueError(
                f"At least {self.timesteps} observations are required to build a sequence window."
            )

        values = reduced_df.values.astype(np.float32)
        windows: List[np.ndarray] = []
        window_positions: List[int] = []
        window_timestamps: List[Optional[pd.Timestamp]] = []

        for start in range(len(values) - self.timesteps + 1):
            end = start + self.timesteps
            windows.append(values[start:end])
            position = end - 1
            index_value = reduced_df.index[position]
            timestamp = index_value if isinstance(index_value, pd.Timestamp) else None
            window_positions.append(int(position))
            window_timestamps.append(timestamp)

        return np.asarray(windows, dtype=np.float32), window_positions, window_timestamps


__all__ = ["LSTMPreprocessor"]


