import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from joblib import dump


def train_rf_model(features, labels):
    df = features.copy()
    y = labels.dropna().astype(int)
    X = df.loc[y.index]

    # Compute class weights to handle imbalance
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weight_dict = {cls: w for cls, w in zip(classes, class_weights)}

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=25,
        min_samples_leaf=3,
        class_weight=class_weight_dict,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Handle binary or single-class edge case
    if len(model.classes_) > 1:
        preds = model.predict_proba(X)[:, 1]  # probability of class 1 (anomaly)
    else:
        preds = np.zeros(len(X))  # fallback for no variation case

    # Ensure model output directory exists
    os.makedirs("models", exist_ok=True)
    dump(model, 'models/rf_model.joblib')

    return preds, model