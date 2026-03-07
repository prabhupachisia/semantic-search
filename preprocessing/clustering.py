import numpy as np
import joblib

from sklearn.mixture import GaussianMixture


MODEL_PATH = "artifacts/cluster_model.pkl"
PROB_PATH = "artifacts/cluster_probs.npy"


def find_best_cluster_count(embeddings):

    best_k = None
    best_bic = float("inf")

    for k in range(10, 40, 5):

        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            random_state=42
        )

        gmm.fit(embeddings)

        bic = gmm.bic(embeddings)

        if bic < best_bic:
            best_bic = bic
            best_k = k

    print("Best cluster count:", best_k)

    return best_k


def run_clustering(embeddings):

    k = find_best_cluster_count(embeddings)

    gmm = GaussianMixture(
        n_components=k,
        covariance_type="diag",
        random_state=42
    )

    gmm.fit(embeddings)

    probabilities = gmm.predict_proba(embeddings)

    joblib.dump(gmm, MODEL_PATH)

    np.save(PROB_PATH, probabilities)

    print("Cluster model saved")

    return gmm, probabilities