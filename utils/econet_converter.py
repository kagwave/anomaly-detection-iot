import pandas as pd
import os
import numpy as np
import warnings

# Suppress divide-by-zero warnings globally
warnings.filterwarnings("ignore", category=RuntimeWarning)

def convert_econet_long_to_wide(input_csv, output_csv):
    """
    Converts EcoNET long-form sensor data to wide-form format compatible with preprocessing pipeline.
    Includes sampling and column filtering for manageable pipeline performance.
    """
    df = pd.read_csv(input_csv, parse_dates=['Ob'])

    # TEMP: Cap to first 2 weeks of data for performance
    df = df[df['Ob'] < '2021-01-15']

    # Create unified sensor_id (e.g., temp_mean_EBO)
    df['sensor_id'] = df['measure'] + '_' + df['Station']

    # Pivot to wide format: each sensor_id becomes a column
    wide = df.pivot_table(index='Ob', columns='sensor_id', values='value', aggfunc='mean')

    # Drop sensors with >50% missing to reduce dimensionality
    wide = wide.loc[:, wide.isna().mean() < 0.5]

    # Derive anomaly label: if any sensor at that timestamp is an anomaly, mark as 1
    if 'target' in df.columns:
        labels = df.groupby('Ob')['target'].max()
        labels = labels.astype(int) if labels.dtype == bool else labels
        wide['anomaly'] = labels

    # Attach station ID back (approximation for visualization)
    wide['station'] = df.groupby('Ob')['Station'].first().values

    wide.reset_index(inplace=True)
    wide.rename(columns={'Ob': 'timestamp'}, inplace=True)

    # Sanitize anomaly labels
    if 'anomaly' in wide.columns:
        wide['anomaly'] = wide['anomaly'].fillna(0).astype(int)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    wide.to_csv(output_csv, index=False)
    print(f"[EcoNET] Converted data saved to {output_csv} with shape {wide.shape}")
    return wide