# Network Intrusion Detection System

An Explainable Machine Learning System for Network Intrusion Detection Using Ensemble Classifiers

## Overview

NIDS is a full-stack network intrusion detection system that uses ensemble machine learning models to classify network traffic as normal or one of 12 attack types. It features a FastAPI backend with SHAP-based explainability and a real-time dark-themed dashboard for monitoring predictions.


##Features
🔍 Multi-class attack detection — classifies traffic into 12 attack categories
🤖 Ensemble ML models — Random Forest + XGBoost for high accuracy
🧠 SHAP explainability — per-prediction feature importance explanations
📊 Live dashboard — Chart.js visualizations with prediction history
🗃️ SQLite persistence — all predictions stored via /history endpoint
⚙️ Custom feature engineering — suspicion score, privilege escalation score, and more
⚖️ SMOTE balancing — handles class imbalance in training data

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the models:
```bash
python IDS_MODEL.py
```

This generates model files and preprocessor objects needed by the API.

3. Start the API:
```bash
python main.py
```

The web interface will be available at `http://localhost:8000`

## Project Structure

- `IDS_MODEL.py` - Model training and evaluation
- `main.py` - FastAPI application and endpoints
- `test_samples.py` - Test script for model predictions
- `nsl-kdd/` - Training and test datasets
- `templates/` - Web interface HTML

## Models

The system uses two ensemble models:
- **Random Forest** - Baseline classifier
- **XGBoost** - Gradient boosting model

Both are trained on NSL-KDD dataset with feature scaling and class balancing.

## Datasets

Training uses NSL-KDD files:
- KDDTrain+_20Percent.txt - Reduced training set
- KDDTrain+.txt - Full training set
- KDDTest+.txt - Test set

## API Endpoints

- `GET /` - Web interface
- `POST /predict` - Get predictions with explanations
- `GET /health` - Check API status


