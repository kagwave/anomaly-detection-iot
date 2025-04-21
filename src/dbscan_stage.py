import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances

def run_dbscan_filter(features_df):
    df = features_df.copy()

    # Determine adaptive eps based on average nearest-neighbor distance (1km proxy)
    distances = pairwise_distances(df, metric='euclidean')
    np.fill_diagonal(distances, np.inf)
    nearest_neighbors = np.min(distances, axis=1)
    adaptive_eps = np.percentile(nearest_neighbors, 30)  # e.g., 30th percentile

    # Adaptive min_samples based on dimensionality (or network degree in real data)
    min_samples = int(np.log(df.shape[1]) * 2)

    db = DBSCAN(eps=adaptive_eps, min_samples=min_samples)
    cluster_labels = db.fit_predict(df)

    # Convert to anomaly likelihood: noise points = -1, others scaled
    anomaly_score = np.where(cluster_labels == -1, 1.0, 0.0)

    df['anomaly_score'] = anomaly_score
    return df, anomaly_score