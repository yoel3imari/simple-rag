import pickle

import numpy as np
import requests

from src.config import METADATA_PATH, OLLAMA_BASE_URL, OLLAMA_GEN_MODEL, TOP_K, VECTORS_PATH
from src.embed import embed


def query(question: str) -> tuple[str, str]:
    if not VECTORS_PATH.exists() or not METADATA_PATH.exists():
        return "No vector store found. Run `ingest` first.", ""

    # 1. Load vectors and metadata
    index_vectors = np.load(str(VECTORS_PATH))
    with open(METADATA_PATH, "rb") as f:
        metadata: list[dict] = pickle.load(f)

    # 2. Embed question
    q_vec = embed(question)[0]

    # 3. Normalize query vector for cosine similarity
    q_vec_norm = q_vec.reshape(1, -1).astype(np.float32)
    q_norm = np.linalg.norm(q_vec_norm)
    q_vec_norm = q_vec_norm / q_norm

    # 4. Search: cosine similarity via dot product on unit vectors
    similarities = np.dot(index_vectors, q_vec_norm.T).flatten()
    top_k_indices = np.argsort(similarities)[-TOP_K:][::-1]
    top_k_scores = similarities[top_k_indices]

    # 5. Build context
    context_parts: list[str] = []
    for score, idx in zip(top_k_scores, top_k_indices):
        meta = metadata[idx] # TODO: what if metadata is empty
        context_parts.append(
            f"[{meta['source']}: chunk {meta['chunk_index']}] (score: {score:.4f})\n{meta['text']}"
        )

    if not context_parts:
        return "No relevant chunks found.", ""

    context_str = "\n\n".join(context_parts)

    # 6. Prompt
    prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided context.
If the context doesn't contain enough information to answer, say "I cannot answer based on the provided documents."

Context:
{context_str}

Question: {question}

Answer:"""

    # 7. Call Ollama
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": OLLAMA_GEN_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    answer = resp.json()["message"]["content"]

    return answer, context_str
