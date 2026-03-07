from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(self, model_name="all-MiniLM-L6-v2"):

        self.model = SentenceTransformer(model_name)

    def encode_documents(self, texts):

        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings

    def encode_query(self, query):

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]

        return embedding