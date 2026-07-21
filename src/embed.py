import numpy as np
import requests

from src.config import OLLAMA_BASE_URL, OLLAMA_EMBEDDING_MODEL


def embed(texts: str | list[str]) -> np.ndarray:
    if isinstance(texts, str):
        texts = [texts]

    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": OLLAMA_EMBEDDING_MODEL, "input": texts},
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()

    return np.array(data["embeddings"], dtype=np.float32)
