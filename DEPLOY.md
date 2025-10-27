# 🚀 Deployment Guide - Streamlit Cloud

## ✅ Pre-Deployment Checklist

### 1. Configured Files
- [x] `app.py` - Loads resources from GitHub
- [x] `requirements.txt` - All dependencies listed
- [x] `.streamlit/config.toml` - Premium dark theme configured
- [x] `.gitignore` - Configured correctly
- [x] `README.md` - Deployment instructions updated

### 2. GitHub URLs Configured
The app is configured to automatically load from the repository:
```
https://github.com/sidnei-almeida/secom_failure_prediction
```

**Files automatically loaded:**
- 📊 `data/secom_cleaned_dataset.csv`
- 🧠 `models/secom_autoencoder_model.keras`
- 📝 `training/secom_autoencoder_metadata.json`

### 3. Required Dependencies
```
✓ streamlit>=1.28.0
✓ streamlit-option-menu>=0.3.6
✓ tensorflow-cpu>=2.15.0
✓ pandas>=2.0.0
✓ numpy>=1.24.0
✓ scikit-learn>=1.3.0
✓ plotly>=5.17.0
✓ Pillow>=10.0.0
✓ requests>=2.31.0
```

## 📤 Deployment Steps

### 1. Commit and Push to GitHub
```bash
# Add all files (including data/, models/, training/)
git add .

# Commit
git commit -m "Deploy: App ready for Streamlit Cloud"

# Push to main
git push origin main
```

### 2. Deploy to Streamlit Cloud

1. Access: [share.streamlit.io](https://share.streamlit.io)
2. Login with GitHub
3. Click "New app"
4. Select:
   - **Repository**: `sidnei-almeida/secom_failure_prediction`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click "Deploy!"

### 3. Wait for Build
Streamlit Cloud will:
- Install dependencies from `requirements.txt`
- Automatically load files from GitHub
- Apply theme from `.streamlit/config.toml`
- Start the app

⏱️ Estimated time: 3-5 minutes

## 🎨 App Features

### Pages
1. **🏠 Home** - Overview and main metrics
2. **📊 Data Analysis** - SECOM dataset exploration
3. **🧠 Model** - Autoencoder architecture
4. **📈 Training** - History and performance
5. **🔬 Test** - Real-time anomaly detection

### Design
- 🌑 Premium dark theme
- 🔥 Hot color palette (orange/fire)
- ✨ Elegant visual effects (glows, shadows)
- 📱 Responsive layout

### Detection Thresholds
- **Balanced (0.45)**: Balance between precision and recall
- **Conservative (0.50)**: Fewer false positives

## 🔧 Troubleshooting

### Error loading data
- Verify files are committed to GitHub
- Confirm repository is public or Streamlit Cloud has access
- Branch must be `main`

### Dependency error
- Check `requirements.txt`
- TensorFlow CPU is used for compatibility

### Theme error
- File `.streamlit/config.toml` must be in the repository
- Must not be in `.gitignore`

## 📞 Support

- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)
- [Streamlit Forum](https://discuss.streamlit.io/)

---

**✨ Ready for deployment!** The app is 100% configured to run on Streamlit Cloud without any additional configuration.
