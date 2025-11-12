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

- 🧠 **Sequence model**: single-layer LSTM (50 hidden units) trained on timesteps of SECOM telemetry.
- 🧪 **Reproducible preprocessing**: Python pipeline mirrors training (gap filling, MinMax scaling, variance + correlation filtering, sliding windows).
- 🎯 **Default classification threshold**: 0.7325 (F1-optimal balance of precision and recall), with request-level overrides.
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

### `POST /predict`
Detect anomalies from sequential SECOM samples.

**Request body**
```json
{
  "instances": [
    [0.12, -0.53, 0.88, "...", 0.34],
    [0.11, -0.51, 0.83, "...", 0.31],
    "...",
    [0.09, -0.47, 0.80, "...", 0.29]
  ],
  "threshold": 0.7325
}
```

- `instances`: chronologically ordered observations. Each observation must contain **590** sensor features and you must supply at least **10** rows to generate the first prediction window.
- `threshold` (optional): override detection threshold; defaults to **0.7325**.
- `timestamps` (optional): ISO datetime strings aligned with each observation.

> ⚠️ Arrays are truncated above for readability. Ensure every observation carries the full feature set.

**Response**
```json
{
  "threshold": 0.7325,
  "feature_names": ["0", "1", "...", "589"],
  "predictions": [
    {
      "window_end_index": 9,
      "timestamp": "2008-07-19T12:32:00",
      "probability": 0.84,
      "is_anomaly": true
    },
    {
      "window_end_index": 10,
      "timestamp": "2008-07-19T12:33:00",
      "probability": 0.21,
      "is_anomaly": false
    }
  ]
}
```

## 📂 Project Structure

```
secom_failure_prediction/
├── main.py                          # FastAPI service
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Hugging Face Space compatible image
├── README.md                        # Project documentation
├── models/
│   ├── lstm_model.keras             # Trained LSTM weights (Git LFS)
│   └── processed_uci_secom.csv      # Processed dataset used to fit preprocessing pipeline
├── preprocess_pipeline.py            # Reusable preprocessing class for LSTM input
└── training/
    └── secom_autoencoder_metadata.json (legacy reference only)
```

## 📊 SECOM Dataset

- **Total Records (original dataset)**: 1,567
- **Features**: 590 (after cleaning, variance filtering, and correlation pruning)
- **Classes**: Normal (-1) vs Failure (1)
- **Class Imbalance**: ≈93% Normal vs 7% Failures

The repository ships only derived assets required for inference. `training/scaler_params.json` stores the StandardScaler statistics computed on the normal subset (`Pass/Fail = -1`), preserving the training regime without bundling the raw CSV.

## 🎓 Methodology

1. **Preprocessing**: Forward/backward fills, MinMax scaling, low-variance filtering, high-correlation pruning, sliding windows of 10 timesteps.
2. **Architecture**: LSTM (50 hidden units) + sigmoid output for failure probability.
3. **Training**: Supervised on the processed SECOM dataset, capturing temporal context.
4. **Inference**: Probability per window; classify as anomaly when probability ≥ threshold (default 0.7325).

## 🧪 Validation

- Precision-Recall sweep identified **0.7325** as the optimal F1 threshold.
- Applying this cut-off yields **Accuracy 96.92%, Precision 73.11%, Recall 84.47%, F1 78.38%** on the validation set.

## 📝 License

This project is licensed under the MIT license.

---

For questions or improvements, feel free to open an issue or pull request. Happy detecting! 🛠️
