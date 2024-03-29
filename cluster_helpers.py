from sklearn.cluster import DBSCAN, KMeans
import numpy as np
from sklearn.metrics import silhouette_score


def get_DBSCAN_cluster_info(state, eps, min_samples):
        db = DBSCAN(eps = eps, min_samples = min_samples).fit(state.reshape(-1,1))
        labels = db.labels_
        n_clusters =  len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        return (labels, n_clusters, n_noise)


def get_KMeans_cluster_info(state, max_clusters, max_iter, tol):
    sil_scores = np.zeros(max_clusters-1)
    for i in range(2, max_clusters+1):
        db = KMeans(n_clusters = i, max_iter = max_iter, tol = tol).fit(state.reshape(-1,1))
        sil_scores[i-2] = silhouette_score(state.reshape(-1,1), db.labels_)
    n_clusters = np.argmax(sil_scores)+2
    db = KMeans(n_clusters = n_clusters, max_iter = max_iter, tol = tol).fit(state.reshape(-1,1))
    idx = np.argsort(db.cluster_centers_.sum(axis=1))
    lut = np.zeros_like(idx)
    lut[idx] = np.arange(n_clusters)
    labels = lut[db.labels_]

    return (labels, n_clusters)