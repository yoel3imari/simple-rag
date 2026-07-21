from pathlib import Path

import quiver_vector_db as quiver
from src.bm25 import UnicodeBM25
from src.prompts import build_generation_prompt

from src.config import (
    BM25_STATE_PATH,
    OLLAMA_GEN_MODEL,
    QUERY_TRANSFORM_MODE,
    QUIVER_STORE_PATH,
    TOP_K,
    TOP_K_RERANK,
)
from src.llm import ollama_chat
from src.query_transform import (
    should_transform,
    detect_language,
    rewrite_query,
    generate_hypothetical_doc,
    run_hyde_fusion,
    _standard_retrieve,
)
from src.constants import TransformMode


def _load_store() -> tuple:
    """Load and return (collection, bm25) from persistent store."""
    client = quiver.Client(path=str(QUIVER_STORE_PATH))
    col = client.get_collection("docs")
    bm25 = UnicodeBM25.load(str(BM25_STATE_PATH))
    return col, bm25


def _prepare_embed_text(question: str, mode: str | TransformMode) -> tuple[str, str | None]:
    """Decide what text to embed based on the transform *mode*.

    Returns ``(embed_text, lang)`` where *lang* may be None if no transform
    was applied (detected lazily downstream).
    """
    embed_text = question
    lang = None
    if mode in (TransformMode.REWRITE, TransformMode.HYDE, TransformMode.HYDE_FUSION) and should_transform(question):
        lang = detect_language(question)
        if mode == TransformMode.REWRITE:
            embed_text = rewrite_query(question, lang)
        elif mode == TransformMode.HYDE:
            embed_text = generate_hypothetical_doc(question, lang)
    return embed_text, lang


def _build_context(hits: list[dict], scores: list[float]) -> str:
    """Build a context string from ranked hits and their relevance scores."""
    parts = []
    for score, hit in zip(scores, hits):
        payload = hit["payload"]
        parts.append(
            f"[{payload['source']}: chunk {payload['chunk_index']}] (relevance: {score:.4f})\n{payload['text']}"
        )
    if not parts:
        return ""
    return "\n\n".join(parts)


def query(question: str, transform_mode: str | None = None) -> tuple[str, str]:
    if not QUIVER_STORE_PATH.exists():
        return "No vector store found. Run `ingest` first.", ""

    col, bm25 = _load_store()

    mode = transform_mode or QUERY_TRANSFORM_MODE
    embed_text, lang = _prepare_embed_text(question, mode)

    if mode == TransformMode.HYDE_FUSION and should_transform(question):
        hits, scores = run_hyde_fusion(
            question, col, bm25,
            top_k=TOP_K, use_rerank=True, use_bm25=True,
        )
    else:
        hits, scores = _standard_retrieve(
            question, embed_text, col, bm25,
            top_k=TOP_K, top_k_rerank=TOP_K_RERANK,
        )

    context_str = _build_context(hits, scores)
    if not context_str:
        return "No relevant chunks found.", ""

    answer = ollama_chat(OLLAMA_GEN_MODEL, build_generation_prompt(question, context_str))
    return answer, context_str
