from pathlib import Path

from src.data_loader import load_deap
from src.preprocessing import remove_baseline
from src.features import extract_bandpower_features
from src.kmeans import (
    prepare_features,
    find_best_k,
    run_kmeans
)
from src.evaluation import summarize_deap_clusters
from src.visualization import (
    plot_elbow,
    plot_clusters_2d
)


# ==========================================================
# CONFIGURATION
# ==========================================================

DATA_DIR = Path("data/deap")

SAMPLING_FREQUENCY = 128
BASELINE_SECONDS = 3


# ==========================================================
# 1. LOAD DATA
# ==========================================================

data, labels, metadata = load_deap(
    DATA_DIR,
    subject_ids=[1]
)

print("\nRaw data shape:")
print(data.shape)

print("\nLabel shape:")
print(labels.shape)


# ==========================================================
# 2. PREPROCESS
# ==========================================================

data = remove_baseline(
    data,
    fs=SAMPLING_FREQUENCY,
    baseline_seconds=BASELINE_SECONDS
)

print("\nAfter baseline removal:")
print(data.shape)


# ==========================================================
# 3. EXTRACT FEATURES
# ==========================================================

features, feature_names = extract_bandpower_features(
    data,
    fs=SAMPLING_FREQUENCY
)

print("\nFeature matrix:")
print(features.shape)

print("\nFeatures:")
print(feature_names)


# ==========================================================
# 4. STANDARDIZATION + PCA
# ==========================================================

X, scaler, pca = prepare_features(features)

print("\nAfter PCA:")
print(X.shape)

print(
    f"Explained variance: "
    f"{pca.explained_variance_ratio_.sum():.2%}"
)


# ==========================================================
# 5. FIND BEST K
# ==========================================================

best_k, k_results = find_best_k(
    X,
    range(2, 11)
)

print(f"\nBest K: {best_k}")


# ==========================================================
# 6. FINAL K-MEANS
# ==========================================================

model, cluster_labels = run_kmeans(
    X,
    n_clusters=best_k
)


# ==========================================================
# 7. EVALUATION
# ==========================================================

summary = summarize_deap_clusters(
    cluster_labels,
    labels
)

print("\nCluster summary:")
print(summary)


# ==========================================================
# 8. VISUALIZATION
# ==========================================================

plot_elbow(k_results)

plot_clusters_2d(
    X,
    cluster_labels
)