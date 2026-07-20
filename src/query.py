from pathlib import Path

import requests

import quiver_vector_db as quiver
from quiver_vector_db import BM25

from src.config import (
    BM25_STATE_PATH,
    HYBRID_DENSE_WEIGHT,
    HYBRID_SPARSE_WEIGHT,
    HYDE_FUSION_TOP_K,
    HYDE_FUSION_RRF_K,
    OLLAMA_BASE_URL,
    OLLAMA_GEN_MODEL,
    QUERY_TRANSFORM_MODE,
    QUIVER_STORE_PATH,
    TOP_K,
    TOP_K_RERANK,
)
from src.embed import embed
from src.rerank import rerank
from src.query_transform import (
    should_transform,
    detect_language,
    rewrite_query,
    generate_hypothetical_doc,
    reciprocal_rank_fusion,
    RAW,
    REWRITE,
    HYDE,
    HYDE_FUSION,
)


def query(question: str, transform_mode: str | None = None) -> tuple[str, str]:
    if not QUIVER_STORE_PATH.exists():
        return "No vector store found. Run `ingest` first.", ""

    # 1. Load Quiver collection and BM25 state
    client = quiver.Client(path=str(QUIVER_STORE_PATH))
    col = client.get_collection("docs")
    bm25 = BM25.load(str(BM25_STATE_PATH))

    # 1.5 Query transformation (v6) — decide what to embed
    mode = transform_mode or QUERY_TRANSFORM_MODE
    embed_text = question  # default: embed raw question

    if mode in (REWRITE, HYDE, HYDE_FUSION) and should_transform(question):
        lang = detect_language(question)
        if mode == REWRITE:
            embed_text = rewrite_query(question, lang)
        elif mode == HYDE:
            embed_text = generate_hypothetical_doc(question, lang)
        # HYDE_FUSION handled in its own block below

    # ── HYDE_FUSION: dual search + RRF fusion ──────────────────
    if mode == HYDE_FUSION and should_transform(question):
        lang = detect_language(question)
        hyde_text = generate_hypothetical_doc(question, lang)

        # Raw branch
        raw_vec = embed(question)[0]
        raw_sparse = bm25.encode_query(question)
        raw_hits = col.search_hybrid(
            dense_query=raw_vec.tolist(),
            sparse_query=raw_sparse,
            k=HYDE_FUSION_TOP_K,
            dense_weight=HYBRID_DENSE_WEIGHT,
            sparse_weight=HYBRID_SPARSE_WEIGHT,
        )
        # HyDE branch
        hyde_vec = embed(hyde_text)[0]
        hyde_sparse = bm25.encode_query(hyde_text)
        hyde_hits = col.search_hybrid(
            dense_query=hyde_vec.tolist(),
            sparse_query=hyde_sparse,
            k=HYDE_FUSION_TOP_K,
            dense_weight=HYBRID_DENSE_WEIGHT,
            sparse_weight=HYBRID_SPARSE_WEIGHT,
        )

        # Fuse by chunk_index via RRF
        raw_rank = [h["payload"]["chunk_index"] for h in raw_hits]
        hyde_rank = [h["payload"]["chunk_index"] for h in hyde_hits]
        fused = reciprocal_rank_fusion([raw_rank, hyde_rank], k=HYDE_FUSION_RRF_K)

        # Rebuild hits list from fused chunk_index order
        by_chunk = {h["payload"]["chunk_index"]: h for h in raw_hits + hyde_hits}
        hits = [by_chunk[ci] for ci, _ in fused if ci in by_chunk][:TOP_K]

        # Rerank with ORIGINAL question
        texts = [h["payload"]["text"] for h in hits]
        ce_scores = rerank(question, texts)
        ranked = sorted(zip(hits, ce_scores), key=lambda x: x[1], reverse=True)
        hits = [h for h, _ in ranked[:TOP_K]]
        scores = [s for _, s in ranked[:TOP_K]]
    else:
        # 2. Embed question (possibly transformed)
        q_vec = embed(embed_text)[0]

        # 3. Hybrid search — fetch larger candidate pool for re-ranking
        sparse_query = bm25.encode_query(question)
        hits = col.search_hybrid(
            dense_query=q_vec.tolist(),
            sparse_query=sparse_query,
            k=TOP_K_RERANK,
            dense_weight=HYBRID_DENSE_WEIGHT,
            sparse_weight=HYBRID_SPARSE_WEIGHT,
        )

        # 4. Cross-encoder re-ranking (ALWAYS with ORIGINAL question)
        texts = [hit["payload"]["text"] for hit in hits]
        ce_scores = rerank(question, texts)
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
