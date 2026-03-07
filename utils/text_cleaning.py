import re


def clean_text(text: str) -> str:
    """
    Basic cleaning for the noisy 20 newsgroups dataset.
    Removes URLs, extra spaces and normalizes case.
    """

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"\n+", " ", text)

    text = re.sub(r"\s+", " ", text)

    text = text.strip()

    return text