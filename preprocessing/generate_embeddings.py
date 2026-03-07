import numpy as np

from utils.embeddings import EmbeddingModel


EMBEDDINGS_PATH = "artifacts/embeddings.npy"


def generate_embeddings(documents):

    model = EmbeddingModel()

    texts = [doc["text"] for doc in documents]

    embeddings = model.encode_documents(texts)

    np.save(EMBEDDINGS_PATH, embeddings)

    print("Embeddings saved to artifacts/embeddings.npy")

    return embeddings