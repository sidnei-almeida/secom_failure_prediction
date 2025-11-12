# SECOM Failure Prediction - Inference API

Advanced anomaly detection service for semiconductor manufacturing, powered by a Neural Network Autoencoder and exposed through a FastAPI HTTP interface.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-ready-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 About the Project

**SECOM Failure Prediction** identifies failures in semiconductor manufacturing processes through anomaly detection. A pretrained **Neural Network Autoencoder** learns normal operating patterns (558 sensor features) and flags deviations using reconstruction error.

### Key Features

- 🧠 **Autoencoder** architecture: 558 → 128 → 64 → 32 (bottleneck) → 64 → 128 → 558
- ⚙️ **FastAPI** service with `/predict`, `/health`, and metadata endpoints
- 🎯 **Threshold control**: default balanced threshold 0.45, override per request
- 🗂️ **Bundled assets**: pretrained autoencoder, training metadata, and scaler statistics
- 🚀 **Hugging Face Space ready**: Dockerfile + `requirements.txt`

### Model Metrics

- **Recall (Anomalies)**: 35.6%
- **Precision (Anomalies)**: 44.6%
- **F1-Score**: 0.396
- **Overall Accuracy**: 71.5%

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
Detect anomalies in one or more SECOM samples.

**Request body**
```json
{
  "instances": [
    [0.12, -0.53, 0.88, 1.07],
    [0.02, 0.11, -0.42, -0.34]
  ],
  "threshold": 0.45
}
```

- `instances`: array of samples, each with 558 numeric sensor readings in the same order as the cleaned dataset.
- `threshold` (optional): override detection threshold; defaults to 0.45.

> ⚠️ The example above shows only four values per sample for brevity. Provide all **558** features when calling the API.

**Response**
```json
{
  "threshold": 0.45,
  "predictions": [
    {"reconstruction_error": 0.38, "is_anomaly": false},
    {"reconstruction_error": 0.57, "is_anomaly": true}
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
│   └── secom_autoencoder_model.keras # Pretrained autoencoder
└── training/
    ├── secom_autoencoder_metadata.json # Training history and metrics
    └── scaler_params.json              # StandardScaler parameters for inference
```

## 📊 SECOM Dataset

- **Total Records (original dataset)**: 1,567
- **Features**: 558 (after cleaning and removing high-missing-value columns)
- **Classes**: Normal (-1) vs Failure (1)
- **Class Imbalance**: ≈93% Normal vs 7% Failures

The repository ships only derived assets required for inference. `training/scaler_params.json` stores the StandardScaler statistics computed on the normal subset (`Pass/Fail = -1`), preserving the training regime without bundling the raw CSV.

## 🎓 Methodology

1. **Preprocessing**: Cleaned features, removed high-missing-value columns, median imputation.
2. **Architecture**: Symmetric autoencoder with 32-dimensional bottleneck.
3. **Training**: Trained exclusively on normal samples (1,170 records).
4. **Detection**: Mean Absolute Error (MAE) between input and reconstruction.
5. **Thresholds**: `0.45` (balanced default) and `0.50` (conservative option).

## 🧪 Validation

Evaluation performed on the held-out validation set captured in `training/secom_autoencoder_metadata.json`. The API reproduces the same inference logic for consistent results.

## 📝 License

This project is licensed under the MIT license.

---

For questions or improvements, feel free to open an issue or pull request. Happy detecting! 🛠️
