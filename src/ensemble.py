import numpy as np
import shap
import pandas as pd
import os
from joblib import dump
import matplotlib.pyplot as plt


def ensemble_predictions(rf_preds, svm_preds, lstm_preds):
    # Normalize all scores to 0–1 if not already
    rf_preds = (rf_preds - np.min(rf_preds)) / (np.max(rf_preds) - np.min(rf_preds) + 1e-8)
    svm_preds = (svm_preds - np.min(svm_preds)) / (np.max(svm_preds) - np.min(svm_preds) + 1e-8)
    lstm_preds = (lstm_preds - np.min(lstm_preds)) / (np.max(lstm_preds) - np.min(lstm_preds) + 1e-8)

    # Confidence-weighted voting
    final_preds = (0.3 * rf_preds) + (0.3 * svm_preds) + (0.4 * lstm_preds)
    return final_preds


def run_shap_analysis(model, X, preds):
    os.makedirs("outputs", exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Save bar summary plot
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("outputs/shap_summary_bar.png")
    plt.close()

    # Save dot summary plot
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig("outputs/shap_summary_dot.png")
    plt.close()

    # Optionally save raw shap values
    np.save("outputs/shap_values.npy", shap_values)
    dump(model, "models/rf_with_shap.joblib")