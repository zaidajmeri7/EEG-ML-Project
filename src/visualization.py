import matplotlib.pyplot as plt


def plot_elbow(results):

    k_values = [r["k"] for r in results]
    scores = [r["silhouette_score"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, scores, marker="o")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette Score vs K")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_clusters_2d(features, cluster_labels):

    plt.figure(figsize=(8, 6))

    plt.scatter(
        features[:, 0],
        features[:, 1],
        c=cluster_labels,
        alpha=0.7
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("K-Means EEG Clusters")
    plt.colorbar(label="Cluster")
    plt.tight_layout()
    plt.show()