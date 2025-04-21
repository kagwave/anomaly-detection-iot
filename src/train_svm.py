import pandas as pd
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from joblib import dump


def train_svm_model(features, labels):
    df = features.copy()
    y = labels.dropna().astype(int)
    X = df.loc[y.index]

    # Only train if there are at least 2 classes
    if len(np.unique(y)) < 2:
        print("[SVM] Skipping training — only one class present in data.")
        return np.zeros(len(X)), None

    # Train/val split (before scaling)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Recheck class count in training subset
    unique_train_classes = np.unique(y_train)
    if len(unique_train_classes) < 2:
        print("[SVM] Skipping training — only one class in training subset.")
        return np.zeros(len(X)), None

    # Normalize features for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_scaled_full = scaler.transform(X)

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=unique_train_classes, y=y_train)
    class_weight_dict = {cls: w for cls, w in zip(unique_train_classes, class_weights)}

    model = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        probability=True,
        class_weight=class_weight_dict,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    preds = model.predict_proba(X_scaled_full)[:, 1]

    os.makedirs("models", exist_ok=True)
    dump(model, 'models/svm_model.joblib')
    dump(scaler, 'models/svm_scaler.joblib')

    return preds, model
