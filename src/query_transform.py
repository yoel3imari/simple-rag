import requests
from collections import defaultdict

from src.config import (
    HYDE_FUSION_RRF_K,
    HYDE_PROMPT_AR,
    HYDE_PROMPT_EN,
    HYDE_PROMPT_FR,
    OLLAMA_BASE_URL,
    OLLAMA_GEN_MODEL,
    REWRITE_PROMPT_AR,
    REWRITE_PROMPT_EN,
    REWRITE_PROMPT_FR,
    TRANSFORM_MIN_ENTITIES,
    TRANSFORM_MIN_LENGTH,
)

# ── Transform modes ──────────────────────────────────────────
RAW = "none"
REWRITE = "rewrite"
HYDE = "hyde"
HYDE_FUSION = "hyde_fusion"


# ── Helpers ──────────────────────────────────────────────────
def ollama_chat(prompt: str) -> str:
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
    return resp.json()["message"]["content"].strip()


# ── Language detection ───────────────────────────────────────
_FRENCH_MARKERS = frozenset(
    {"le", "la", "les", "une", "que", "qui", "est", "pour", "dans", "avec"}
)


def detect_language(text: str) -> str:
    """Return ``"ar"``, ``"fr"``, or ``"en"`` based on heuristic markers."""
    # Arabic Unicode block
    if any("\u0600" <= c <= "\u06FF" for c in text):
        return "ar"

    # French stopwords — require at least 2 to avoid English false positives
    lower = text.lower()
    words = lower.split()
    french_hits = sum(1 for w in words if w in _FRENCH_MARKERS)
    if french_hits >= 2:
        return "fr"

    return "en"


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
def rewrite_query(question: str, lang: str = "en") -> str:
    prompt_map = {"en": REWRITE_PROMPT_EN, "ar": REWRITE_PROMPT_AR, "fr": REWRITE_PROMPT_FR}
    template = prompt_map.get(lang, REWRITE_PROMPT_EN)
    prompt = template.format(question=question)
    return ollama_chat(prompt)


# ── HyDE: hypothetical document generation ──────────────────
def generate_hypothetical_doc(question: str, lang: str = "en") -> str:
    prompt_map = {"en": HYDE_PROMPT_EN, "ar": HYDE_PROMPT_AR, "fr": HYDE_PROMPT_FR}
    template = prompt_map.get(lang, HYDE_PROMPT_EN)
    prompt = template.format(question=question)
    return ollama_chat(prompt)


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
