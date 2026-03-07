import numpy as np
import faiss

from collections import defaultdict
from app.config import config


class SemanticCache:

    def __init__(self):

        self.cache = {}

        self.hit_count = 0
        self.miss_count = 0

    def _init_cluster(self, cluster_id):

        dim = 384

        index = faiss.IndexFlatIP(dim)

        self.cache[cluster_id] = {
            "index": index,
            "embeddings": [],
            "entries": []
        }

    def lookup(self, query_embedding, cluster_id):

        if cluster_id not in self.cache:

            self.miss_count += 1
            return None

        cluster = self.cache[cluster_id]

        if len(cluster["entries"]) == 0:

            self.miss_count += 1
            return None

        query_vec = np.array([query_embedding]).astype("float32")

        scores, ids = cluster["index"].search(query_vec, 1)

        score = scores[0][0]
        idx = ids[0][0]

        if score >= config.similarity_threshold:

            self.hit_count += 1

            return cluster["entries"][idx], score

        self.miss_count += 1

        return None

    def add(self, query, embedding, result, cluster_id):

        if cluster_id not in self.cache:

            self._init_cluster(cluster_id)

        cluster = self.cache[cluster_id]

        vec = np.array([embedding]).astype("float32")

        cluster["index"].add(vec)

        cluster["embeddings"].append(embedding)

        cluster["entries"].append({
            "query": query,
            "result": result
        })

    def clear(self):

        self.cache.clear()

        self.hit_count = 0
        self.miss_count = 0

    def stats(self):

        total_entries = sum(
            len(cluster["entries"])
            for cluster in self.cache.values()
        )

        total = self.hit_count + self.miss_count

        return {
            "total_entries": total_entries,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": self.hit_count / total if total else 0
        }


cache = SemanticCache()