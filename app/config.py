import json
import faiss
import joblib

from utils.embeddings import EmbeddingModel


class AppConfig:

    def __init__(self):

        self.embedder = EmbeddingModel()

        self.index = faiss.read_index("artifacts/faiss.index")

        self.cluster_model = joblib.load("artifacts/cluster_model.pkl")

        with open("artifacts/documents.json") as f:
            self.documents = json.load(f)

        self.similarity_threshold = 0.85


config = AppConfig()