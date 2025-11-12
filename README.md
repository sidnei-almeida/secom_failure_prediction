---
title: Secom Production Anomaly
emoji: 📚
colorFrom: green
colorTo: gray
sdk: docker
pinned: false
license: mit
---

# SECOM Failure Prediction - Inference API

Production-ready anomaly detection service for semiconductor manufacturing, powered by an LSTM sequence classifier and exposed through a FastAPI HTTP interface.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-ready-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 About the Project

**SECOM Failure Prediction** identifies failures in semiconductor manufacturing processes through anomaly detection. A pretrained **LSTM classifier** analyses rolling windows of SECOM sensor readings (10 timesteps × 590 features) and estimates the probability of a failure at the next step.

### Key Features

- 🧠 **Sequence model**: single-layer LSTM (50 hidden units) trained on 10-step SECOM telemetry windows.
- 🔄 **Online preprocessing endpoint**: converts raw readings (timestamp + 590 features) to Min-Max scaled values using saved training statistics.
- 📈 **Rolling predictions**: `/predict` buffers the latest readings and responds with a probability once the 10-sample window is complete (resettable per simulation run).
- 🎯 **Default classification threshold**: 0.7325 (F1-optimal balance), overridable per request.
- 📊 **Rich metadata**: `/` endpoint reports feature set, timesteps, and evaluation metrics.
- 🚀 **Hugging Face Space ready**: Dockerfile + `requirements.txt`, Git LFS tracking for large artifacts.

### Model Metrics (threshold = 0.7325)

- **Accuracy**: 96.92%
- **Precision**: 73.11%
- **Recall**: 84.47%
- **F1-Score**: 78.38%

## 🚀 Run Locally

```bash
git clone https://github.com/sidnei-almeida/secom_failure_prediction.git
cd secom_failure_prediction
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```

The API will be available at `http://localhost:7860`.

## 🐳 Docker

```bash
docker build -t secom-api .
docker run -p 7860:7860 secom-api
```

## 🌐 Hugging Face Space

1. Create a new **Space** using the **Docker** runtime.
2. Clone the Space repo locally and enable Git LFS (Spaces enforce a 15 GB soft limit and require LFS for files >10 MB):
   ```bash
   git clone https://huggingface.co/spaces/<org>/<space_name>
   cd <space_name>
   git lfs install --system
   ```
3. Copy the project files into the Space working tree. The `.gitattributes` in this repo already tracks model binaries with LFS.
4. Commit and push (Hugging Face rejects commits with blobs >50 MB that are not in LFS):
   ```bash
   git add .
   git commit -m "Add SECOM Failure Prediction API"
   git push
   ```
5. Hugging Face will build the Docker image and expose the API at `/` once the push succeeds.

Optional: use `huggingface_hub` to automate pushes (`huggingface-cli login` first) or to host the model/dataset in a separate model repo.

## 📡 API Reference

### `GET /`
Returns project metadata, model type, default threshold, and evaluation metrics.

### `GET /health`
Simple health probe used by deployment platforms.

### `POST /preprocess`
Applies Min-Max scaling to a single raw reading.

**Request body**
```json
{
  "reading": {
    "timestamp": "2008-07-19T12:32:00Z",
    "values": [0.12, -0.53, 0.88, "...", 0.34]
  }
}
```

- `values`: one observation containing **590** sensor features.
- `timestamp` (optional): ISO datetime string; echoed back after validation.

**Response**
```json
{
  "scaled_values": [0.48, 0.32, 0.66, "...", 0.55],
  "timestamp": "2008-07-19T12:32:00+00:00"
}
```

### `POST /predict`
Ingests a reading, updates the rolling buffer, and returns a prediction once ten readings are available.

**Request body**
```json
{
  "reading": {
    "timestamp": "2008-07-19T12:32:00Z",
    "values": [0.12, -0.53, 0.88, "...", 0.34]
  },
  "threshold": 0.7325,
  "reset_buffer": false
}
```

- `reading`: same structure as `/preprocess`.
- `threshold` (optional): override the classification threshold for this request.
- `reset_buffer` (optional): clear the rolling window before ingesting the reading (useful when starting a new simulation run).

**Response (after the 10th reading)**
```json
{
  "scaled_values": [0.48, 0.32, 0.66, "...", 0.55],
  "buffer_size": 10,
  "timesteps": 10,
  "prediction": {
    "probability": 0.84,
    "is_anomaly": true,
    "threshold": 0.7325,
    "window_end_timestamp": "2008-07-19T12:32:00+00:00"
  }
}
```

When fewer than ten readings have been ingested, the `prediction` field is `null`, allowing the frontend to keep streaming simulated data until the window fills.

## 📂 Project Structure

```
secom_failure_prediction/
├── main.py                          # FastAPI service
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Hugging Face Space compatible image
├── README.md                        # Project documentation
├── models/
│   ├── lstm_model.keras             # Trained LSTM weights (Git LFS)
│   └── minmax_stats.json            # Min/Max statistics captured during training
└── preprocess_pipeline.py            # Online scaler and rolling window helpers
```

## 📊 SECOM Dataset

- **Total Records (original dataset)**: 1,567
- **Features**: 590 (after cleaning, variance filtering, and correlation pruning)
- **Classes**: Normal (-1) vs Failure (1)
- **Class Imbalance**: ≈93% Normal vs 7% Failures

The repository ships only derived assets required for inference. `models/minmax_stats.json` stores the Min-Max statistics captured during training, so the API can scale each incoming reading without shipping the raw CSV.

## 🎓 Methodology

1. **Preprocessing (training)**: Gap filling, Min-Max scaling, variance/correlation filtering, sliding windows of 10 timesteps.
2. **Online preprocessing**: Validate timestamp, scale features with saved Min-Max stats, update internal rolling window.
3. **Architecture**: LSTM (50 hidden units) + sigmoid output for failure probability.
4. **Inference**: Probability per window; classify as anomaly when probability ≥ threshold (default 0.7325).

## 🧪 Validation

- Precision-Recall sweep identified **0.7325** as the optimal F1 threshold.
- Applying this cut-off yields **Accuracy 96.92%, Precision 73.11%, Recall 84.47%, F1 78.38%** on the validation set.

## 📝 License

This project is licensed under the MIT license.

---

For questions or improvements, feel free to open an issue or pull request. Happy detecting! 🛠️
