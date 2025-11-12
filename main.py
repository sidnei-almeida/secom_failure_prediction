from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from preprocess_pipeline import LSTMOnlinePreprocessor, PreprocessingConfig, WindowManager

PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "models" / "lstm_model.keras"
PREPROCESSING_PATH = PROJECT_ROOT / "models" / "minmax_stats.json"
DEFAULT_THRESHOLD = 0.7325
EVALUATION_METRICS = {
    "accuracy": 0.9692,
    "precision": 0.7311,
    "recall": 0.8447,
    "f1": 0.7838,
}


class SensorReading(BaseModel):
    values: List[float] = Field(..., description="Raw sensor vector (590 features).")
    timestamp: Optional[str] = Field(None, description="ISO timestamp associated with this reading.")


class PreprocessRequest(BaseModel):
    reading: SensorReading


class PreprocessResponse(BaseModel):
    scaled_values: List[float]
    timestamp: Optional[str]


class PredictRequest(BaseModel):
    reading: SensorReading
    threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Optional override for the classification threshold.",
    )
    reset_buffer: bool = Field(
        False,
        description="Set to true to clear the internal sequence buffer before ingesting this reading.",
    )


class PredictionResult(BaseModel):
    probability: float = Field(..., description="Predicted failure probability for the current window.")
    is_anomaly: bool = Field(..., description="True when probability exceeds the chosen threshold.")
    threshold: float = Field(..., description="Threshold used for this prediction.")
    window_end_timestamp: Optional[str] = Field(
        None, description="Timestamp associated with the last observation in the prediction window."
    )


class PredictResponse(BaseModel):
    scaled_values: List[float]
    buffer_size: int
    timesteps: int
    prediction: Optional[PredictionResult]


def make_app() -> FastAPI:
    config = PreprocessingConfig.load(PREPROCESSING_PATH)
    preprocessor = LSTMOnlinePreprocessor(config)
    window_manager = WindowManager(config.timesteps)
    model = tf.keras.models.load_model(MODEL_PATH)

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
            "feature_count": len(preprocessor.feature_names),
            "feature_names": preprocessor.feature_names,
            "default_threshold": DEFAULT_THRESHOLD,
            "metrics": EVALUATION_METRICS,
        }

    @app.get("/health", summary="Health check")
    def health():
        return {"status": "ok"}

    @app.post("/preprocess", response_model=PreprocessResponse, summary="Scale a sensor reading")
    def preprocess(request: PreprocessRequest):
        try:
            scaled = preprocessor.scale(request.reading.values)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return PreprocessResponse(
            scaled_values=scaled.tolist(),
            timestamp=request.reading.timestamp,
        )

    @app.post("/predict", response_model=PredictResponse, summary="Ingest a reading and predict with the LSTM")
    def predict(request: PredictRequest):
        if request.reset_buffer:
            window_manager.reset()

        try:
            scaled = preprocessor.scale(request.reading.values)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        sequence, window_timestamp = window_manager.add(scaled, request.reading.timestamp)

        prediction_payload: Optional[PredictionResult] = None
        buffer_size = window_manager.size()
        used_threshold = request.threshold if request.threshold is not None else DEFAULT_THRESHOLD

        if sequence is not None:
            window_batch = sequence[np.newaxis, ...]
            probability = float(model.predict(window_batch, verbose=0)[0][0])
            prediction_payload = PredictionResult(
                probability=probability,
                is_anomaly=bool(probability >= used_threshold),
                threshold=float(used_threshold),
                window_end_timestamp=window_timestamp,
            )

        return PredictResponse(
            scaled_values=scaled.tolist(),
            buffer_size=buffer_size,
            timesteps=preprocessor.timesteps,
            prediction=prediction_payload,
        )

    return app


app = make_app()



