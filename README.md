# AI-Based DoS/DDoS Attack Detection System

A machine learning-based network traffic classification system that identifies benign and potentially malicious DoS/DDoS traffic using a Random Forest classifier.

## Overview

This project uses the CIC-IDS2017 dataset to train a machine learning model for classifying network traffic into two categories:

- BENIGN
- ATTACK

It also provides a Streamlit dashboard for offline traffic analysis and live network traffic monitoring.

## Features

- Machine learning-based network traffic classification.
- Random Forest classifier trained on CIC-IDS2017.
- CSV upload and traffic analysis.
- Live packet capture using Scapy.
- Network flow feature extraction.
- Live prediction and monitoring dashboard.

## Technology Stack

- Python
- Scikit-learn
- Pandas and NumPy
- Scapy
- Streamlit
- Joblib
- CIC-IDS2017 dataset

## Model Features

The classifier uses the following eight network flow features:

1. Flow Duration
2. Total Fwd Packets
3. Total Backward Packets
4. Total Length of Fwd Packets
5. Total Length of Bwd Packets
6. Flow Bytes/s
7. Flow Packets/s
8. Packet Length Mean

## Project Architecture

1. **Data Input:** Network traffic is provided through a CSV dataset or live packet capture.
2. **Feature Extraction:** Network traffic is converted into numerical flow-level features.
3. **Preprocessing:** Required features are validated and missing numerical values are handled by the model pipeline.
4. **Classification:** The trained Random Forest model predicts BENIGN or ATTACK.
5. **Dashboard:** Streamlit displays traffic analysis and prediction results.

## Project Structure

```text
dos-detection-system/
│
├── app.py
├── models/
│   └── cicids2017_validated_model.pkl
│
├── src/
│   ├── capture_live_traffic.py
│   ├── extract_live_features.py
│   ├── predict_live_features.py
│   └── train_validated_model.py
│
├── requirements.txt
├── README.md
└── .gitignore
