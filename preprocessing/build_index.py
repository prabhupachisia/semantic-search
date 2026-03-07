import faiss
import numpy as np


INDEX_PATH = "artifacts/faiss.index"


def build_index(embeddings):

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)

    index.add(embeddings.astype("float32"))

    faiss.write_index(index, INDEX_PATH)

    print("FAISS index saved")

    return index