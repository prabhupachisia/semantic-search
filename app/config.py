import json
import faiss
import joblib
from pathlib import Path

from utils.embeddings import EmbeddingModel


BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BASE_DIR / "artifacts"


class AppConfig:

    def __init__(self):

        self.embedder = EmbeddingModel()

        self.index = faiss.read_index(str(ARTIFACT_DIR / "faiss.index"))

        self.cluster_model = joblib.load(ARTIFACT_DIR / "cluster_model.pkl")

        with open(ARTIFACT_DIR / "documents.json") as f:
            docs = json.load(f)

        self.documents = [
            {"text": doc["text"][:200]}
            for doc in docs
        ]

        self.similarity_threshold = 0.85

config = AppConfig()