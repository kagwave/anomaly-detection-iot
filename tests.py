"""
Minimal test script to validate that each stage of the pipeline works on a small mock dataset.
"""

import numpy as np
import pandas as pd
from src.preprocess import preprocess_data
from src.features import extract_features
from src.dbscan_stage import run_dbscan_filter
from src.train_rf import train_rf_model
from src.train_svm import train_svm_model
from src.train_lstm import train_lstm_model
from src.ensemble import ensemble_predictions, run_shap_analysis
from utils.metrics import evaluate_models


def generate_mock_data():
    timestamps = pd.date_range("2023-01-01", periods=200, freq="5min")
    mock_data = {
        'timestamp': timestamps,
        'sensor_A': np.sin(np.linspace(0, 20, 200)) + np.random.normal(0, 0.1, 200),
        'sensor_B': np.random.normal(25, 3, 200),
        'sensor_C': np.random.normal(100, 10, 200),
        'anomaly': [0]*190 + [1]*10  # few anomalies at the end
    }
    return pd.DataFrame(mock_data)


def test_pipeline():
    df = generate_mock_data()
    df.to_csv("data/raw/mock_sensor_data.csv", index=False)

    df_processed = preprocess_data("data/raw/mock_sensor_data.csv", "data/processed/mock_processed.csv")
    features, labels = extract_features(df_processed)

    filtered, _ = run_dbscan_filter(features)
    rf_preds, rf_model = train_rf_model(filtered, labels)
    svm_preds, svm_model = train_svm_model(filtered, labels)
    lstm_preds, lstm_model = train_lstm_model(filtered, labels)

    final_preds = ensemble_predictions(rf_preds, svm_preds, lstm_preds)
    run_shap_analysis(rf_model, filtered, final_preds)
    evaluate_models(labels, rf_preds, svm_preds, lstm_preds, final_preds)


if __name__ == "__main__":
    test_pipeline()