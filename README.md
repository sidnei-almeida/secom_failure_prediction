# SECOM Failure Prediction - Anomaly Detection System

Advanced anomaly detection system for semiconductor manufacturing using Neural Network Autoencoder.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 About the Project

**SECOM Failure Prediction** is an anomaly detection system developed to identify failures in semiconductor manufacturing processes. Using a **Neural Network Autoencoder**, the system learns normal operation patterns and detects deviations that may indicate potential failures.

### Key Features

- 🧠 **Neural Network Autoencoder** with architecture 558 → 128 → 64 → 32 (bottleneck) → 64 → 128 → 558
- 📊 **Interactive Dashboard** developed with Streamlit
- 🎯 **Two Detection Thresholds**: Balanced (0.45) and Conservative (0.50)
- 📈 **Advanced Visualizations** with Plotly for data and results analysis
- 🎨 **Premium Dark Design** with hot color palette (industrial/fire)
- ⚡ **Optimized Performance** using TensorFlow CPU

### Model Metrics

- **Recall (Anomalies)**: 35.6%
- **Precision (Anomalies)**: 44.6%
- **F1-Score**: 0.396
- **Overall Accuracy**: 71.5%

## 🌐 Deploy to Streamlit Cloud

The application is **fully configured** for deployment to Streamlit Cloud! Data and models are automatically loaded from GitHub.

### How to deploy:

1. **Push code to GitHub** (including `data/`, `models/`, `training/` folders):
```bash
git add .
git commit -m "Deploy ready"
git push origin main
```

2. **Access** [share.streamlit.io](https://share.streamlit.io)
3. **Connect** your GitHub repository
4. **Select** main file: `app.py`
5. **Automatic deploy!** 🚀

The app will automatically load all necessary resources from GitHub.

## 🚀 Run Locally

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/sidnei-almeida/secom_failure_prediction.git
cd secom_failure_prediction
```

2. **Create and activate a virtual environment** (recommended)
```bash
python -m venv venv

# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

The application will automatically open in your browser at `http://localhost:8501`

## 📂 Project Structure

```
secom_failure_prediction/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Project dependencies
├── README.md                       # This file
├── data/
│   └── secom_cleaned_dataset.csv  # Cleaned dataset (1567 records, 558 features)
├── models/
│   └── secom_autoencoder_model.keras  # Trained model
├── training/
│   └── secom_autoencoder_metadata.json  # Training metadata
└── notebooks/
    ├── 1_Data_Analysis_and_Manipulation.ipynb
    ├── 2_Deep_Learning_Models_Classification.ipynb
    └── 3_Anomaly_Detection.ipynb
```

## 🎯 App Features

### 1. **Home**
- Project overview and main metrics
- Class distribution (Normal vs Failures)
- Key insights about dataset and methodology

### 2. **Data Analysis**
- Descriptive statistics of features
- Distribution visualizations
- Correlation matrix
- Interactive SECOM dataset exploration

### 3. **Model**
- Detailed explanation of Autoencoder architecture
- Interactive neural network visualization
- Anomaly detection process description
- Complete technical specifications

### 4. **Training**
- Complete training history
- Loss evolution graphs (training and validation)
- Final performance metrics
- Configurations and hyperparameters used

### 5. **Test**
- CSV file upload for testing
- Threshold selection (Balanced or Conservative)
- Real-time analysis with visualizations
- Reconstruction error distribution
- Confusion matrix (when labels are available)
- Download results as CSV

## 🧪 Testing the System

You can test the system using the project's own dataset:

1. Go to the **Test** page
2. Upload the file `data/secom_cleaned_dataset.csv`
3. Select desired threshold
4. Click **Analyze**
5. View results and download report

## 🛠️ Technologies Used

- **TensorFlow/Keras**: Deep Learning framework
- **Streamlit**: Dashboard framework
- **Plotly**: Interactive visualization library
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Preprocessing and metrics

## 📊 SECOM Dataset

The SECOM dataset contains sensor data from a semiconductor manufacturing process:

- **Total Records**: 1567
- **Features**: 558 (after cleaning and removal of features with >40% missing values)
- **Classes**: Binary (Normal: -1, Failure: 1)
- **Imbalance**: ~93% Normal vs ~7% Failures

## 🎓 Methodology

1. **Preprocessing**: Data cleaning, removal of features with excess null values, median imputation
2. **Architecture**: Symmetric autoencoder with 32-dimension bottleneck
3. **Training**: Only with normal data (1170 samples)
4. **Detection**: Reconstruction error (MAE) > threshold = anomaly
5. **Thresholds**: 
   - **Balanced (0.45)**: Best precision-recall balance
   - **Conservative (0.50)**: Fewer false positives

## 📝 License

This project is licensed under the MIT license.

## 👨‍💻 Author

Developed with ❤️ for advanced anomaly analysis in industrial processes.

---

**Note**: This is an academic/professional project developed to demonstrate Deep Learning techniques applied to anomaly detection in industrial environments.
