import os
from src.preprocess import preprocess_data
from src.features import extract_features
from src.dbscan_stage import run_dbscan_filter
from src.train_rf import train_rf_model
from src.train_svm import train_svm_model
from src.train_lstm import train_lstm_model
from src.ensemble import ensemble_predictions, run_shap_analysis
from utils.metrics import evaluate_models
from utils.econet_converter import convert_econet_long_to_wide
from utils.zone_overlay import plot_zone_anomalies_from_train
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def main():
    # Step 0: Convert EcoNET dataset (long → wide)
    if not os.path.exists("data/raw/smart_city_iot.csv"):
        convert_econet_long_to_wide("data/raw/train.csv", "data/raw/smart_city_iot.csv")

    # Step 1: Preprocess raw data
    print("Preprocessing raw data...")
    raw_path = "data/raw/smart_city_iot.csv"
    processed_path = "data/processed/processed_data.csv"
    df_clean = preprocess_data(raw_path, processed_path)

    # Step 2: Feature Engineering
    print("Extracting features...")
    feature_df, full_labels = extract_features(df_clean)

    # Subsample for speed
    subset_df = feature_df.iloc[:5000]
    subset_labels = full_labels.loc[subset_df.index]  # <-- align here first

    # Step 3: DBSCAN
    print("Running DBSCAN anomaly filtering...")
    filtered_df, _ = run_dbscan_filter(subset_df)

    # Align labels after DBSCAN filtering
    aligned_labels = subset_labels.loc[filtered_df.index]

    print("Checking index consistency...")
    print("filtered_df:", filtered_df.shape, "index[0]:", filtered_df.index[0])
    print("aligned_labels:", aligned_labels.shape, "index[0]:", aligned_labels.index[0])
    print("Index match:", all(filtered_df.index == aligned_labels.index))
    print("Label preview:", aligned_labels.head())

    # Step 4: Supervised classification
    print("Training Random Forest...")
    rf_preds, rf_model = train_rf_model(filtered_df, aligned_labels)
    print("Training SVM...")
    svm_preds, svm_model = train_svm_model(filtered_df, aligned_labels)
    print("Training LSTM...")
    lstm_preds, lstm_model = train_lstm_model(filtered_df, aligned_labels)

    # Step 5: Ensemble and SHAP-based interpretation
    print("Running ensemble and SHAP...")
    final_preds = ensemble_predictions(rf_preds, svm_preds, lstm_preds)
    run_shap_analysis(rf_model, filtered_df, final_preds)

    # Step 6: Evaluation
    print("Labels shape:", aligned_labels.shape)
    print("RF preds:", rf_preds.shape)
    print("SVM preds:", svm_preds.shape)
    print("LSTM preds:", lstm_preds.shape)
    print("Ensemble preds:", final_preds.shape)

    assert len(aligned_labels) == len(rf_preds), f"Label/pred mismatch: {len(aligned_labels)} vs {len(rf_preds)}"

    print("Evaluating models...")
    evaluate_models(aligned_labels, rf_preds, svm_preds, lstm_preds, final_preds)

    # Step 7: Visualization
    print("[EcoNET] Generating zone anomaly map...")
    plot_zone_anomalies_from_train()

if __name__ == "__main__":
    main()