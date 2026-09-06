import numpy as np
import pandas as pd


def summarize_deap_clusters(
    cluster_labels: np.ndarray,
    emotion_labels: np.ndarray
) -> pd.DataFrame:

    results = pd.DataFrame({
        "cluster": cluster_labels,
        "valence": emotion_labels[:, 0],
        "arousal": emotion_labels[:, 1],
        "dominance": emotion_labels[:, 2],
        "liking": emotion_labels[:, 3],
    })

    summary = (
        results
        .groupby("cluster")
        .agg({
            "valence": "mean",
            "arousal": "mean",
            "dominance": "mean",
            "liking": "mean"
        })
    )

    return summary