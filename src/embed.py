import numpy as np

from src.config import OLLAMA_EMBEDDING_MODEL
from src.llm import ollama_embed as _ollama_embed


def embed(texts: str | list[str]) -> np.ndarray:
    if isinstance(texts, str):
        texts = [texts]
    return _ollama_embed(OLLAMA_EMBEDDING_MODEL, texts)
