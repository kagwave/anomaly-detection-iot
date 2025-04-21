import pandas as pd
import numpy as np
from scipy.fft import fft
from statsmodels.tsa.stattools import adfuller
from datetime import datetime

def extract_features(df):
    df = df.copy()

    # Ensure timestamp index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    df.set_index('timestamp', inplace=True)

    label_col = 'anomaly' if 'anomaly' in df.columns else None
    labels_raw = df[label_col] if label_col else pd.Series(index=df.index, data=0)

    feature_list = []

    for col in df.columns:
        if col == label_col:
            continue

        # Rolling stats
        rolling_15min = df[col].rolling(window=3).agg(['mean', 'std'])
        rolling_1hr = df[col].rolling(window=12).agg(['mean', 'std'])

        # FFT (global, static per column)
        fft_vals = np.abs(fft(df[col].fillna(0)))[:5]
        fft_cols = {f'{col}_fft_{i}': fft_vals[i] for i in range(5)}

        # ADF p-values
        adf_p = []
        for i in range(len(df[col])):
            window = df[col].iloc[max(0, i-10):i+1].dropna()
            if len(window) > 5:
                try:
                    adf_pval = adfuller(window, autolag='AIC')[1]
                except:
                    adf_pval = 1.0
            else:
                adf_pval = 1.0
            adf_p.append(adf_pval)

        combined = pd.concat([rolling_15min, rolling_1hr], axis=1)
        combined.columns = [f'{col}_15min_mean', f'{col}_15min_std', f'{col}_1hr_mean', f'{col}_1hr_std']
        combined[f'{col}_adf_p'] = adf_p
        for k, v in fft_cols.items():
            combined[k] = v

        feature_list.append(combined)

    # Contextual features
    df_ctx = pd.DataFrame(index=df.index)
    df_ctx['hour'] = df_ctx.index.hour
    df_ctx['dayofweek'] = df_ctx.index.dayofweek
    df_ctx['is_weekend'] = df_ctx['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

    # Concatenate all features
    full_features = pd.concat(feature_list + [df_ctx], axis=1)

    # Drop rows with any NaNs (affects both features + labels)
    full_features = full_features.dropna()
    labels = labels_raw.reindex(full_features.index).fillna(0).astype(int)

    return full_features, labels


    