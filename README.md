# Anomaly Detection in Smart City IoT Data

This repository implements a modular machine learning pipeline for detecting, ranking, and interpreting anomalies in IoT sensor streams, specifically tailored for smart city infrastructure. The pipeline is designed to integrate multi-stage filtering, interpretable supervised learning, and SHAP-based feature attribution for explainable results.

## Objective

The aim is to surface actionable anomalies from heterogeneous, irregularly-sampled time series originating from weather and infrastructure sensors. Our system supports scalable ingest of real-world IoT datasets (e.g., EcoNET), identifies significant deviations from learned baselines, and enables interpretation through model-based explanations.

## Pipeline Architecture

### 1. Data Ingestion & Preprocessing
- Converts long-form EcoNET sensor logs to wide-form matrices with per-sensor columns
- Resamples to uniform 5-minute intervals
- Imputes missing values using KNN with spatial-aware neighbors
- Applies z-score normalization using rolling 1-day window

### 2. Feature Engineering
- Rolling temporal statistics (mean, std)
- Spectral power features from FFT
- Change point detection via Augmented Dickey-Fuller (ADF) test
- Temporal context: hour-of-day, weekday/weekend, etc.

### 3. Unsupervised Outlier Filtering
- Uses DBSCAN with dynamic \( \varepsilon \) and log-scaled `min_samples`
- Filters extreme sparse outliers before supervised learning
- Adaptively chooses threshold from distance distributions

### 4. Supervised Anomaly Classifiers
- **Random Forest**: interpretable tree-based voting model with class weights
- **SVM (RBF kernel)**: margin-based classifier with scaling and balancing
- **Bidirectional LSTM**: sequential model for long-range temporal dependencies

### 5. Ensemble Decision
- Weighted soft-voting ensemble using calibrated outputs
- Emphasizes models with sharper confidence under high class imbalance

### 6. Interpretability Layer
- Applies SHAP (TreeExplainer) to Random Forest classifier
- Outputs both global and local feature importance via bar/dot plots
- Stores model artifacts + SHAP values in structured output folders

## Evaluation Protocol
- Binary classification metrics:
  - F1 Score (macro)
  - ROC-AUC (if available)
  - Precision at 90% recall
  - Overall Accuracy
- Output summary saved to terminal and logs

## Usage Instructions

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Convert EcoNET Data (One-time)
```bash
python3 -c "from utils.econet_converter import convert_econet_long_to_wide; convert_econet_long_to_wide('data/raw/train.csv', 'data/raw/smart_city_iot.csv')"
```

### 3. Run Full Pipeline
```bash
python main.py
```

### 4. Results Location
- SHAP: `outputs/shap_summary_*.png`
- Models: `models/*.joblib`
- Metrics: printed to stdout

## Notes
- Developed as part of CSC 522 (NCSU) — Spring 2025
- Dataset source: EcoNET (NC State Climate Office)
- Wide-form input must include `timestamp`, `sensor_id`, and `anomaly` columns
- To accelerate DBSCAN: limit rows in `main.py` (e.g., `.iloc[:5000]`)

## Contributors
- Kagwe Muchane
- Jonah Gloss
- Dian Rajic
