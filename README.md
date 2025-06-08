# Fraud Detection & Anomaly Analytics Engine

## Overview

This project focuses on identifying fraudulent credit card transactions using unsupervised machine learning. It combines advanced anomaly detection techniques like Isolation Forest, Autoencoders, and Local Outlier Factor to detect rare events without prior labels. Designed for real-time alerting and explainability, this system can be deployed in modern financial fraud pipelines.

## Key Features

- Multiple anomaly detection models:
  - Isolation Forest
  - Local Outlier Factor
  - Deep Autoencoder
- Unified risk scoring system combining outputs from all models.
- Evaluation using confusion matrix, F1-score, recall, and precision.
- Exportable model predictions for real-time use cases.
- Interactive Streamlit dashboard with filters and fraud summaries.

## Dataset

- **Source**: Kaggle Credit Card Fraud Detection dataset
- **Size**: 284,807 transactions, with only 492 labeled frauds
- **Link**: [Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Project Structure

fraud-detection-anomaly-analytics/
├── app/
│ └── dashboard.py
├── models/
│ ├── isolation_forest_model.pkl
│ ├── autoencoder_model.h5
│ └── scaler.pkl
├── data/
│ └── creditcard.csv "Due to size limits, please download the dataset from [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud]"
│ └── final_fraud_predictions.csv "Not uploaded due to size limits"
├── notebooks/
| └── Fraud_Detection\*&\_Anomaly_Analytics_Engine.ipynb
├── requirements.txt
└── README.md

## Tools & Tech

- Python: `pandas`, `numpy`, `scikit-learn`, `keras`, `matplotlib`, `seaborn`
- Models: Isolation Forest, LOF, Autoencoder
- Dashboard: `Streamlit`
- Deployment: GitHub + Streamlit Cloud

## Results

- Detected over 70% of frauds with Autoencoder (F1 ~0.78)
- Combined risk scoring gives enhanced precision
- All models evaluated for performance and business readiness

## How to Use

streamlit run app/dashboard.py

Make sure the `models/` and `data/` folders are in the same directory.

## Author

Thamizhvaanan Ilango – [Portfolio](https://www.thamizhvaananilango.com)
"# Fraud-Detection-Anomaly-Analytics-Engine"
"# Fraud-Detection-Anomaly-Analytics-Engine" 
