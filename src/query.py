from pathlib import Path

import requests

import quiver_vector_db as quiver
from quiver_vector_db import BM25

from src.config import (
    BM25_STATE_PATH,
    HYBRID_DENSE_WEIGHT,
    HYBRID_SPARSE_WEIGHT,
    OLLAMA_BASE_URL,
    OLLAMA_GEN_MODEL,
    QUIVER_STORE_PATH,
    TOP_K,
    TOP_K_RERANK,
)
from src.embed import embed
from src.rerank import rerank


def query(question: str) -> tuple[str, str]:
    if not QUIVER_STORE_PATH.exists():
        return "No vector store found. Run `ingest` first.", ""

    # 1. Load Quiver collection and BM25 state
    client = quiver.Client(path=str(QUIVER_STORE_PATH))
    col = client.get_collection("docs")
    bm25 = BM25.load(str(BM25_STATE_PATH))

    # 2. Embed question
    q_vec = embed(question)[0]

    # 3. Hybrid search — fetch larger candidate pool for re-ranking
    sparse_query = bm25.encode_query(question)
    hits = col.search_hybrid(
        dense_query=q_vec.tolist(),
        sparse_query=sparse_query,
        k=TOP_K_RERANK,
        dense_weight=HYBRID_DENSE_WEIGHT,
        sparse_weight=HYBRID_SPARSE_WEIGHT,
    )

    # 4. Cross-encoder re-ranking
    texts = [hit["payload"]["text"] for hit in hits]
    ce_scores = rerank(question, texts)
    # Re-sort hits by cross-encoder score descending
    ranked = sorted(zip(hits, ce_scores), key=lambda x: x[1], reverse=True)
    hits = [h for h, _ in ranked[:TOP_K]]
    scores = [s for _, s in ranked[:TOP_K]]

    # 5. Build context from re-ranked results
    context_parts: list[str] = []
    for score, hit in zip(scores, hits):
        payload = hit["payload"]
        context_parts.append(
            f"[{payload['source']}: chunk {payload['chunk_index']}] (relevance: {score:.4f})\n{payload['text']}"
        )

    if not context_parts:
        return "No relevant chunks found.", ""

    context_str = "\n\n".join(context_parts)

    # 6. Prompt
    prompt = f"""You are a helpful assistant. Use the context below to answer the question.

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
