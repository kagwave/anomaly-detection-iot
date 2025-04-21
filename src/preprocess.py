import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


def preprocess_data(raw_path, output_path):
    df = pd.read_csv(raw_path, parse_dates=['timestamp'])

    # Sample equal parts anomalies and normals if label exists
    if 'anomaly' in df.columns:
        anom = df[df['anomaly'] == 1]
        norm = df[df['anomaly'] == 0]

        df_anom = anom.sample(n=min(500, len(anom)), random_state=42)
        df_norm = norm.sample(n=min(1500, len(norm)), random_state=42)

        df = pd.concat([df_anom, df_norm]).sort_values('timestamp')

    # Set timestamp as index
    df.set_index('timestamp', inplace=True)

    # Separate labels if present
    label_col = 'anomaly' if 'anomaly' in df.columns else None
    labels = df[label_col] if label_col else None

    # Drop label and keep only numeric features
    drop_cols = [label_col] if label_col else []
    features = df.drop(columns=drop_cols)
    features = features.select_dtypes(include=[np.number])


    # Interpolate missing timestamps
    start = features.index.min()
    end = features.index.max()
    all_times = pd.date_range(start=start, end=end, freq='5min')
    features = features.reindex(all_times)
    features = features.dropna(axis=1, how='all')  # Drop fully empty sensor columns

    # KNN imputation
    imputer = KNNImputer(n_neighbors=5)
    imputed = pd.DataFrame(imputer.fit_transform(features), index=features.index, columns=features.columns)

    # TEMP: reduce rolling window to 1 day (288 5-min samples)
    rolling_mean = imputed.rolling(window=288, min_periods=50).mean()
    rolling_std = imputed.rolling(window=288, min_periods=50).std()
    normalized = (imputed - rolling_mean) / (rolling_std + 1e-6)

    # Reattach label if it was present
    if label_col:
        normalized[label_col] = labels.reindex(normalized.index)

    normalized.to_csv(output_path)
    return normalized.reset_index().rename(columns={'index': 'timestamp'})
