from sentence_transformers import SentenceTransformer
import torch


class EmbeddingModel:

    def __init__(self):

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.model.eval()

        torch.set_grad_enabled(False)

    def encode_query(self, query):

        embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )[0]

        return embedding