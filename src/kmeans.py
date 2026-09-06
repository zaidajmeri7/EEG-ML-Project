import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def prepare_features(
    features: np.ndarray,
    variance_to_keep: float = 0.95
):
    """
    Standardize features and apply PCA.
    """

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    pca = PCA(n_components=variance_to_keep)
    transformed = pca.fit_transform(scaled)

    return transformed, scaler, pca


def find_best_k(
    features: np.ndarray,
    k_range=range(2, 11)
):
    """
    Find K using silhouette score.
    """

    results = []

    for k in k_range:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )

        cluster_labels = model.fit_predict(features)

        score = silhouette_score(
            features,
            cluster_labels
        )

        results.append({
            "k": k,
            "silhouette_score": score
        })

        print(
            f"K={k}, "
            f"Silhouette={score:.4f}"
        )

    best = max(
        results,
        key=lambda x: x["silhouette_score"]
    )

    return best["k"], results


def run_kmeans(
    features: np.ndarray,
    n_clusters: int
):
    """
    Run final K-Means clustering.
    """

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=20
    )

    labels = model.fit_predict(features)

    return model, labels