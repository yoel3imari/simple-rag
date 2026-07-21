from collections import defaultdict

from src.bm25 import UnicodeBM25
from src.config import (
    HYBRID_DENSE_WEIGHT,
    HYBRID_SPARSE_WEIGHT,
    HYDE_FUSION_TOP_K,
    HYDE_FUSION_RRF_K,
    HYDE_PROMPT_AR,
    HYDE_PROMPT_EN,
    HYDE_PROMPT_FR,
    OLLAMA_GEN_MODEL,
    REWRITE_PROMPT_AR,
    REWRITE_PROMPT_EN,
    REWRITE_PROMPT_FR,
    TOP_K,
    TOP_K_RERANK,
    TRANSFORM_MIN_ENTITIES,
    TRANSFORM_MIN_LENGTH,
)
from src.constants import Language, TransformMode
from src.embed import embed
from src.llm import ollama_chat


# ── Language detection ───────────────────────────────────────
_FRENCH_MARKERS = frozenset(
    {"le", "la", "les", "une", "que", "qui", "est", "pour", "dans", "avec"}
)


def detect_language(text: str) -> Language:
    """Return ``Language.AR``, ``Language.FR``, or ``Language.EN`` based on heuristic markers."""
    # Arabic Unicode block
    if any("\u0600" <= c <= "\u06FF" for c in text):
        return Language.AR

    # French stopwords — require at least 2 to avoid English false positives
    lower = text.lower()
    words = lower.split()
    french_hits = sum(1 for w in words if w in _FRENCH_MARKERS)
    if french_hits >= 2:
        return Language.FR

    return Language.EN


# ── Heuristic gate ───────────────────────────────────────────
def should_transform(question: str) -> bool:
    """Return ``True`` when the query should be transformed.

    Well-formed queries (≥*TRANSFORM_MIN_LENGTH* words **and**
    ≥*TRANSFORM_MIN_ENTITIES* capitalised words) skip transformation
    to avoid regression.
    """
    words = question.split()
    if len(words) < TRANSFORM_MIN_LENGTH:
        return True
    entities = sum(1 for w in words if w[:1].isupper() and len(w) > 1)
    if entities < TRANSFORM_MIN_ENTITIES:
        return True
    return False


# ── Query rewriting ──────────────────────────────────────────
def rewrite_query(question: str, lang: Language = Language.EN) -> str:
    prompt_map = {Language.EN: REWRITE_PROMPT_EN, Language.AR: REWRITE_PROMPT_AR, Language.FR: REWRITE_PROMPT_FR}
    template = prompt_map.get(lang, REWRITE_PROMPT_EN)
    prompt = template.format(question=question)
    return ollama_chat(OLLAMA_GEN_MODEL, prompt)


# ── HyDE: hypothetical document generation ──────────────────
def generate_hypothetical_doc(question: str, lang: Language = Language.EN) -> str:
    prompt_map = {Language.EN: HYDE_PROMPT_EN, Language.AR: HYDE_PROMPT_AR, Language.FR: HYDE_PROMPT_FR}
    template = prompt_map.get(lang, HYDE_PROMPT_EN)
    prompt = template.format(question=question)
    return ollama_chat(OLLAMA_GEN_MODEL, prompt)


# ── Reciprocal Rank Fusion ───────────────────────────────────
def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    k: int | None = None,
) -> list[tuple[int, float]]:
    """Fuse multiple ranked lists of chunk indices via RRF.

    Returns ``(chunk_index, rrf_score)`` sorted descending by score.
    """
    if k is None:
        k = HYDE_FUSION_RRF_K
    scores: dict[int, float] = defaultdict(float)
    for rank_list in ranked_lists:
        for rank, doc_id in enumerate(rank_list, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── HyDE Fusion (dual search + RRF) ──────────────────────────
def run_hyde_fusion(
    question: str,
    col,
    bm25: UnicodeBM25,
    *,
    top_k: int = TOP_K,
    use_rerank: bool = True,
    use_bm25: bool = True,
) -> tuple[list[dict], list[float]]:
    """Dual-search HyDE fusion with RRF + optional cross-encoder reranking.

    Args:
        question: Original user question.
        col: Quiver Collection with hybrid search.
        bm25: UnicodeBM25 instance.
        top_k: Number of final results.
        use_rerank: Apply cross-encoder reranking.
        use_bm25: If False, sets sparse weight to 0.

    Returns:
        (hits, scores) tuple.
    """
    from src.rerank import rerank  # local import to avoid circular deps

    lang = detect_language(question)
    hyde_text = generate_hypothetical_doc(question, lang)

    dense_w = HYBRID_DENSE_WEIGHT
    sparse_w = HYBRID_SPARSE_WEIGHT
    if not use_bm25:
        sparse_w = 0.0

    # Raw branch
    raw_vec = embed(question)[0]
    raw_sparse = bm25.encode_query(question)
    raw_hits = col.search_hybrid(
        dense_query=raw_vec.tolist(),
        sparse_query=raw_sparse,
        k=HYDE_FUSION_TOP_K,
        dense_weight=dense_w,
        sparse_weight=sparse_w,
    )
    # HyDE branch
    hyde_vec = embed(hyde_text)[0]
    hyde_sparse = bm25.encode_query(hyde_text)
    hyde_hits = col.search_hybrid(
        dense_query=hyde_vec.tolist(),
        sparse_query=hyde_sparse,
        k=HYDE_FUSION_TOP_K,
        dense_weight=dense_w,
        sparse_weight=sparse_w,
    )

    # Fuse by chunk_index via RRF
    raw_rank = [h["payload"]["chunk_index"] for h in raw_hits]
    hyde_rank = [h["payload"]["chunk_index"] for h in hyde_hits]
    fused = reciprocal_rank_fusion([raw_rank, hyde_rank], k=HYDE_FUSION_RRF_K)

    by_chunk = {h["payload"]["chunk_index"]: h for h in raw_hits + hyde_hits}
    ranked_hits = [by_chunk[ci] for ci, _ in fused if ci in by_chunk]

    # Rerank with ORIGINAL question (skip for non-English)
    if use_rerank and lang not in (Language.AR, Language.FR) and ranked_hits:
        texts = [h["payload"]["text"] for h in ranked_hits]
        ce_scores = rerank(question, texts)
        sorted_pairs = sorted(zip(ranked_hits, ce_scores), key=lambda x: x[1], reverse=True)
        hits = [h for h, _ in sorted_pairs[:top_k]]
        scores = [s for _, s in sorted_pairs[:top_k]]
    else:
        hits = ranked_hits[:top_k]
        scores = [h.get("score", 0.0) for h in hits]

    return hits, scores


# ── Standard retrieve (embed → hybrid search → rerank) ──────────
def _standard_retrieve(
    question: str,
    embed_text: str,
    col,
    bm25: UnicodeBM25,
    *,
    top_k: int = TOP_K,
    top_k_rerank: int = TOP_K_RERANK,
    use_rerank: bool = True,
    use_bm25: bool = True,
) -> tuple[list[dict], list[float]]:
    """Standard retrieval pipeline: embed → hybrid search → rerank.

    Args:
        question: Original question (used for reranking and BM25 encoding).
        embed_text: Text to embed (may differ from question for rewrite/HyDE).
        col: Quiver Collection.
        bm25: UnicodeBM25 instance.
        top_k: Final result count.
        top_k_rerank: Candidate pool size before reranking (must be >= top_k).
        use_rerank: Apply cross-encoder reranking.
        use_bm25: If False, sets sparse weight to 0 (dense-only).

    Returns:
        (hits, scores) tuple.
    """
    from src.rerank import rerank  # local import to avoid circular deps

    q_vec = embed(embed_text)[0]

    dense_w = HYBRID_DENSE_WEIGHT
    sparse_w = HYBRID_SPARSE_WEIGHT
    if not use_bm25:
        sparse_w = 0.0

    # Determine rerank eligibility (skip non-English)
    if use_rerank:
        lang = detect_language(question)
        if lang in (Language.AR, Language.FR):
            use_rerank = False

    k = top_k_rerank if use_rerank else top_k
    sparse_query = bm25.encode_query(question)
    hits = col.search_hybrid(
        dense_query=q_vec.tolist(),
        sparse_query=sparse_query,
        k=k,
        dense_weight=dense_w,
        sparse_weight=sparse_w,
    )

    if use_rerank and hits:
        texts = [h["payload"]["text"] for h in hits]
        ce_scores = rerank(question, texts)
        ranked = sorted(zip(hits, ce_scores), key=lambda x: x[1], reverse=True)
        ranked_hits = [h for h, _ in ranked[:top_k]]
        scores = [s for _, s in ranked[:top_k]]
    else:
        ranked_hits = hits[:top_k]
        scores = [h.get("score", 0.0) for h in hits]

    return ranked_hits, scores
