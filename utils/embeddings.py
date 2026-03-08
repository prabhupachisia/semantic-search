from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:

    def __init__(self):

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def encode_documents(self, texts):

        embeddings = self.model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True
        )

        return np.array(embeddings)

    def encode_query(self, query):

        embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )[0]

        return np.array(embedding)