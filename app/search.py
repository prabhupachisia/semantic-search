import numpy as np

from app.config import config
from app.cache import cache


def search_documents(query_embedding):

    query_vec = np.array([query_embedding]).astype("float32")

    scores, ids = config.index.search(query_vec, k=5)

    results = []

    for score, idx in zip(scores[0], ids[0]):

        doc = config.documents[idx]

        results.append({
            "text": doc["text"],
            "score": float(score)
        })

    return results


def process_query(query_text):

    query_embedding = config.embedder.encode_query(query_text)

    cluster_probs = config.cluster_model.predict_proba([query_embedding])[0]

    cluster_id = int(np.argmax(cluster_probs))

    cluster_probability = float(cluster_probs[cluster_id])

    cache_result = cache.lookup(query_embedding, cluster_id)

    if cache_result:

        entry, similarity = cache_result

        return {
            "query": query_text,
            "cache_hit": True,
            "matched_query": entry["query"],
            "similarity_score": float(similarity),
            "dominant_cluster": cluster_id,
            "cluster_probability": cluster_probability,
            "result": entry["result"]
        }

    results = search_documents(query_embedding)

    cache.add(
        query_text,
        query_embedding,
        results,
        cluster_id
    )

    return {
        "query": query_text,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "dominant_cluster": cluster_id,
        "cluster_probability": cluster_probability,
        "result": results
    }