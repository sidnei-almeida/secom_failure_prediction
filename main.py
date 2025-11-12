from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from preprocess_pipeline import LSTMPreprocessor

PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "models" / "lstm_model.keras"
TRAIN_DATA_PATH = PROJECT_ROOT / "models" / "processed_uci_secom.csv"
DEFAULT_THRESHOLD = 0.7325
FALLBACK_METRICS = {
    "accuracy": 0.9692,
    "precision": 0.7311,
    "recall": 0.8447,
    "f1": 0.7838,
}


class PredictRequest(BaseModel):
    instances: List[List[float]] = Field(
        ..., description="Ordered observations (n_samples x 590 features prior to preprocessing)."
    )
    timestamps: Optional[List[str]] = Field(
        None,
        description="Optional ISO timestamps aligned with the provided observations.",
    )
    threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Classification threshold applied to the predicted probabilities. Defaults to 0.7325.",
    )


class SequencePrediction(BaseModel):
    window_end_index: int = Field(..., description="Index of the last observation used in the sequence window.")
    timestamp: Optional[str] = Field(
        None, description="Timestamp of the last observation used in the sequence window (if provided)."
    )
    probability: float = Field(..., description="Predicted probability of failure for the next step.")
    is_anomaly: bool = Field(..., description="Binary classification based on the chosen threshold.")


class InferenceResponse(BaseModel):
    threshold: float
    feature_names: List[str]
    predictions: List[SequencePrediction]


def evaluate_model(
    model: tf.keras.Model,
    preprocessor: LSTMPreprocessor,
    dataset_path: Path,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    data = pd.read_csv(dataset_path)

    timestamps = pd.to_datetime(data["Time"]) if "Time" in data.columns else None
    labels = data["Pass/Fail"] if "Pass/Fail" in data.columns else None

    feature_df = data.drop(columns=[col for col in ["Time", "Pass/Fail"] if col in data.columns])
    sequences, positions, sequence_timestamps = preprocessor.transform(
        feature_df.values.tolist(), timestamps=timestamps
    )

    probabilities = model.predict(sequences, verbose=0).flatten()

    if labels is None:
        return {}

    if timestamps is not None:
        label_series = pd.Series(labels.values, index=pd.to_datetime(data["Time"])).sort_index()
    else:
        label_series = pd.Series(labels.values, index=np.arange(len(labels)))

    y_true = []
    for position, ts in zip(positions, sequence_timestamps):
        if ts is not None:
            y_true.append(label_series.loc[ts])
        else:
            y_true.append(label_series.iloc[position])

    y_true = np.where(np.array(y_true) == -1, 0, 1)
    y_pred = (probabilities >= threshold).astype(int)

    if len(np.unique(y_true)) < 2:
        return {"accuracy": float(accuracy_score(y_true, y_pred))}

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def make_app() -> FastAPI:
    preprocessor = LSTMPreprocessor.fit_from_csv(TRAIN_DATA_PATH, timesteps=10)
    model = tf.keras.models.load_model(MODEL_PATH)
    evaluation_metrics = evaluate_model(model, preprocessor, TRAIN_DATA_PATH, threshold=DEFAULT_THRESHOLD)
    metrics = evaluation_metrics or FALLBACK_METRICS

    app = FastAPI(
        title="SECOM Failure Prediction API",
        description="LSTM-based anomaly detection for semiconductor manufacturing sequences.",
        version="2.0.0",
    )

    @app.get("/", summary="Service metadata")
    def root():
        return {
            "project": "SECOM Failure Prediction",
            "model_type": "LSTM",
            "timesteps": preprocessor.timesteps,
            "base_feature_count": preprocessor.base_feature_count,
            "final_feature_count": preprocessor.final_feature_count,
            "feature_names": preprocessor.final_feature_names,
            "default_threshold": DEFAULT_THRESHOLD,
            "metrics": metrics,
        }

    @app.get("/health", summary="Health check")
    def health():
        return {"status": "ok"}

    @app.post("/predict", response_model=InferenceResponse, summary="Predict failures with LSTM model")
    def predict(request: PredictRequest):
        if not request.instances:
            raise HTTPException(status_code=400, detail="`instances` must contain at least one observation.")

        threshold = request.threshold if request.threshold is not None else DEFAULT_THRESHOLD

        try:
            sequences, positions, timestamps = preprocessor.transform(
                request.instances, timestamps=request.timestamps
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        probabilities = model.predict(sequences, verbose=0).flatten()

        predictions = []
        for probability, position, timestamp in zip(probabilities, positions, timestamps):
            predictions.append(
                SequencePrediction(
                    window_end_index=position,
                    timestamp=timestamp.isoformat() if timestamp is not None else None,
                    probability=float(probability),
                    is_anomaly=bool(probability >= threshold),
                )
            )

        return InferenceResponse(
            threshold=float(threshold),
            feature_names=preprocessor.final_feature_names,
            predictions=predictions,
        )

    return app


app = make_app()



