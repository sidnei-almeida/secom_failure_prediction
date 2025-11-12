from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, root_validator
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "models" / "secom_autoencoder_model.keras"
DATASET_PATH = PROJECT_ROOT / "data" / "secom_cleaned_dataset.csv"
METADATA_PATH = PROJECT_ROOT / "training" / "secom_autoencoder_metadata.json"

DEFAULT_THRESHOLD = 0.45
PASS_FAIL_COLUMN = "Pass/Fail"
TIME_COLUMN = "Time"
NUM_FEATURES = 558


class InferenceRequest(BaseModel):
    """
    Input payload for the anomaly detection endpoint.
    """

    instances: List[List[float]] = Field(..., description="Array of samples, each with 558 sensor values.")
    threshold: Optional[float] = Field(
        None,
        ge=0,
        description="Optional override for the anomaly detection threshold. Defaults to 0.45 when omitted.",
    )

    @root_validator
    def _validate_instances(cls, values):
        instances = values.get("instances", [])
        if not instances:
            raise ValueError("`instances` must contain at least one sample.")

        for idx, sample in enumerate(instances):
            if len(sample) != NUM_FEATURES:
                raise ValueError(
                    f"Sample at index {idx} has {len(sample)} features but {NUM_FEATURES} features are required."
                )
        return values


class SamplePrediction(BaseModel):
    reconstruction_error: float = Field(..., description="Mean Absolute Error between original and reconstruction.")
    is_anomaly: bool = Field(..., description="True when reconstruction error is greater than the selected threshold.")


class InferenceResponse(BaseModel):
    threshold: float
    predictions: List[SamplePrediction]


def _load_metadata() -> dict:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}")
    with METADATA_PATH.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found at {DATASET_PATH}")
    return pd.read_csv(DATASET_PATH)


def _load_model() -> tf.keras.Model:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)


def _build_scaler(df: pd.DataFrame) -> StandardScaler:
    feature_df = df.drop(columns=[col for col in [TIME_COLUMN, PASS_FAIL_COLUMN] if col in df.columns], errors="ignore")
    scaler = StandardScaler()
    if PASS_FAIL_COLUMN in df.columns:
        normal_mask = df[PASS_FAIL_COLUMN] == -1
        if not normal_mask.any():
            raise ValueError("Dataset does not contain normal samples (-1) required to fit the scaler.")
        scaler.fit(feature_df.loc[normal_mask])
    else:
        scaler.fit(feature_df)
    return scaler


def make_app() -> FastAPI:
    metadata = _load_metadata()
    dataset = _load_dataset()
    model = _load_model()
    scaler = _build_scaler(dataset)

    feature_df = dataset.drop(columns=[col for col in [TIME_COLUMN, PASS_FAIL_COLUMN] if col in dataset.columns], errors="ignore")
    global NUM_FEATURES
    NUM_FEATURES = feature_df.shape[1]

    app = FastAPI(
        title="SECOM Failure Prediction API",
        description="Neural network autoencoder for anomaly detection in semiconductor manufacturing.",
        version="1.0.0",
    )

    @app.get("/", summary="Service metadata")
    def root():
        performance = metadata.get("final_performance_on_test_set", {})
        return {
            "project": metadata.get("project_name", "SECOM Failure Prediction"),
            "model_type": metadata.get("model_type", "Autoencoder"),
            "default_threshold": metadata.get("final_anomaly_threshold", DEFAULT_THRESHOLD),
            "metrics": {
                "precision_anomaly": performance.get("precision_for_anomaly"),
                "recall_anomaly": performance.get("recall_for_anomaly"),
                "f1_anomaly": performance.get("f1_score_for_anomaly"),
                "accuracy": performance.get("accuracy"),
            },
            "features": NUM_FEATURES,
        }

    @app.get("/health", summary="Health check")
    def health():
        return {"status": "ok"}

    @app.post("/predict", response_model=InferenceResponse, summary="Detect anomalies")
    def predict(request: InferenceRequest):
        threshold = request.threshold if request.threshold is not None else metadata.get(
            "final_anomaly_threshold", DEFAULT_THRESHOLD
        )

        try:
            data = np.asarray(request.instances, dtype=np.float32)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid numeric payload.") from exc

        if data.ndim != 2 or data.shape[1] != NUM_FEATURES:
            raise HTTPException(
                status_code=400,
                detail=f"Each sample must contain {NUM_FEATURES} features. Received shape {data.shape}.",
            )

        try:
            scaled = scaler.transform(data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        reconstructions = model.predict(scaled, verbose=0)
        reconstruction_errors = np.mean(np.abs(reconstructions - scaled), axis=1)
        anomalies = reconstruction_errors > threshold

        predictions = [
            SamplePrediction(reconstruction_error=float(err), is_anomaly=bool(flag))
            for err, flag in zip(reconstruction_errors, anomalies)
        ]

        return InferenceResponse(threshold=float(threshold), predictions=predictions)

    return app


app = make_app()


