import json

from .load_dataset import load_dataset
from .generate_embeddings import generate_embeddings
from .build_index import build_index
from .clustering import run_clustering


def run():

    print("Loading dataset...")
    documents = load_dataset()

    print("Generating embeddings...")
    embeddings = generate_embeddings(documents)

    print("Building FAISS index...")
    build_index(embeddings)

    print("Running fuzzy clustering...")
    gmm, probs = run_clustering(embeddings)

    metadata = {
        "num_documents": len(documents),
        "embedding_dim": embeddings.shape[1],
        "num_clusters": probs.shape[1]
    }

    with open("artifacts/metadata.json", "w") as f:
        json.dump(metadata, f)

    print("Pipeline complete")


if __name__ == "__main__":
    run()