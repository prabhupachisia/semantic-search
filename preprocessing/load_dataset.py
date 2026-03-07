import json
import os

from sklearn.datasets import fetch_20newsgroups

from utils.text_cleaning import clean_text


ARTIFACT_PATH = "artifacts/documents.json"


def load_dataset():

    dataset = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes")
    )

    documents = []

    for i, text in enumerate(dataset.data):

        cleaned = clean_text(text)

        if len(cleaned) < 50:
            continue

        documents.append({
            "id": i,
            "text": cleaned
        })

    os.makedirs("artifacts", exist_ok=True)

    with open(ARTIFACT_PATH, "w") as f:
        json.dump(documents, f)

    print(f"Saved {len(documents)} cleaned documents")

    return documents