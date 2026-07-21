"""Unicode-aware BM25 subclass that fixes the ASCII-only tokenizer in Quiver's BM25.

The stock BM25 uses ``_TOKEN_RE = re.compile(r"[a-z0-9]+")`` which produces
zero tokens for Arabic text and mangles accented French.  ``UnicodeBM25``
overrides ``index_document`` and ``encode_query`` to use a Unicode tokenizer
(with NFKD normalization, combining-mark stripping, and a broader regex)
while keeping all BM25 math identical to the parent.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Dict

from quiver_vector_db import BM25

# ---------------------------------------------------------------------------
# Unicode tokenizer
# ---------------------------------------------------------------------------

_UNI_TOKEN_RE = re.compile(
    r"[\u0600-\u06FF"       # Arabic
    r"\u0750-\u077F"        # Arabic Supplement
    r"\u08A0-\u08FF"        # Arabic Extended-A
    r"\u00C0-\u024F"        # Latin Extended (accented chars)
    r"a-z0-9"               # ASCII alphanumeric (lowercased)
    r"]+"
)

# Comprehensive stop word lists for English, Arabic, and French.
# Sourced from widely-used IR stop word lists (SMART/NLTK for EN,
# standard AR linguistics lists for AR, common FR IR lists for FR).
# These are filtered out during BM25 tokenization since their IDF is
# near-zero and they add only noise to sparse retrieval.
_UNI_STOPWORDS = frozenset({
    # ── English (~125 words) ─────────────────────────────────────
    "a", "about", "above", "after", "again", "against", "all", "am",
    "an", "and", "any", "are", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by",
    "can", "cannot", "could", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has",
    "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "if", "in", "into", "is", "it", "its",
    "itself", "just", "me", "might", "more", "most", "my", "myself",
    "no", "nor", "not", "now", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own",
    "per", "quite", "s", "same", "shall", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was",
    "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "will", "with", "would", "you", "your", "yours",
    "yourself",
    # ── Arabic (~65 words) ────────────────────────────────────────
    "في", "من", "على", "أن", "إن", "هذا", "هذه", "مع", "عن",
    "إلى", "التي", "الذي", "الذين", "اللاتي", "اللواتي", "اللذين",
    "كان", "كانت", "كانوا", "كن", "لم", "لن", "لا", "ما", "هل",
    "قد", "لقد", "إذا", "إذ", "حين", "حتى", "عند", "بين", "تحت",
    "فوق", "أمام", "خلف", "دون", "قبل", "بعد", "هو", "هي", "هم",
    "هن", "أنت", "أنتم", "أنتن", "نحن", "أنا", "ذلك", "تلك",
    "هؤلاء", "هناك", "هنا", "كل", "بعض", "أي", "أو", "ثم", "غير",
    "نحو", "لقد", "ولقد", "إنما", "أما", "سوف",
    # ── French (~100 words) ───────────────────────────────────────
    "le", "la", "les", "des", "une", "pour", "dans", "avec",
    "que", "qui", "est", "sont", "par", "sur", "pas", "au", "aux",
    "ce", "ces", "cet", "cette", "de", "du", "et", "en", "il",
    "elle", "elles", "ils", "je", "tu", "nous", "vous", "mais",
    "ou", "où", "comme", "si", "un", "mon", "ton", "son", "mes",
    "tes", "ses", "nos", "vos", "leurs", "lui", "eux", "leur",
    "ceci", "cela", "ça", "plus", "moins", "très", "peu", "assez",
    "trop", "donc", "car", "ni", "ne", "se", "me", "te", "sous",
    "avant", "après", "pendant", "depuis", "jusque", "hors", "chez",
    "vers", "sans", "sauf", "selon", "malgré", "dès", "dans",
    "entre", "parmi", "depuis", "voici", "voilà", "même", "tous",
    "tout", "toute", "toutes", "quel", "quelle", "quels", "quelles",
    "chaque", "plusieurs", "certains", "certaines",
})


def _uni_tokenize(text: str) -> list[str]:
    """Unicode-aware tokenizer: NFKD normalize, strip combining marks,
    lowercase, extract tokens via Unicode regex, remove stopwords, min
    length 2.
    """
    # Decompose (é → e + combining acute) then drop combining marks
    norm = unicodedata.normalize("NFKD", text)
    norm = "".join(c for c in norm if not unicodedata.combining(c))
    norm = norm.lower()
    tokens = _UNI_TOKEN_RE.findall(norm)
    return [t for t in tokens if len(t) >= 2 and t not in _UNI_STOPWORDS]


# ---------------------------------------------------------------------------
# UnicodeBM25
# ---------------------------------------------------------------------------


class UnicodeBM25(BM25):
    """BM25 subclass with a Unicode-aware tokenizer.

    Overrides ``index_document`` and ``encode_query`` (the two public methods
    that call the module-level ``_tokenize`` in the parent) to use
    ``_uni_tokenize`` instead.  All BM25 math (idf, tf, avgdl), instance
    attributes, and persistence (``save``/``load``) are inherited unchanged.
    """

    def index_document(self, doc_id: int, text: str) -> Dict[int, float]:
        """Tokenize text with Unicode-aware tokenizer, update IDF stats,
        return BM25 sparse vector.

        Args:
            doc_id: The vector ID (same as used with ``upsert_hybrid``).
            text: Raw document text to index.

        Returns:
            Sparse vector as ``{dimension_index: bm25_weight}`` suitable for
            ``Collection.upsert_hybrid(sparse_vector=...)``.
            Empty dict if text produces no tokens.
        """
        tokens = _uni_tokenize(text)
        if not tokens:
            return {}

        # If re-indexing, remove old stats first
        if doc_id in self._doc_lengths:
            old_dl = self._doc_lengths[doc_id]
            self._doc_count -= 1
            self._total_dl -= old_dl

        tf_counts = Counter(tokens)
        dl = len(tokens)

        # Update document frequency for each unique term
        for token in tf_counts:
            tid = self._get_or_create_term_id(token)
            self._df[tid] = self._df.get(tid, 0) + 1

        # Update global stats
        self._doc_count += 1
        self._total_dl += dl
        self._doc_lengths[doc_id] = dl

        # Compute BM25 weights
        avgdl = self.avg_dl
        sparse: Dict[int, float] = {}
        for token, tf in tf_counts.items():
            tid = self._vocab[token]
            df = self._df.get(tid, 0)
            idf = math.log((self._doc_count - df + 0.5) / (df + 0.5) + 1.0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
            weight = idf * numerator / denominator
            if weight > 0:
                sparse[tid] = weight

        return sparse

    def encode_query(self, text: str) -> Dict[int, float]:
        """Encode a query into an IDF-weighted sparse vector.

        Only uses terms already in the vocabulary -- unseen terms are ignored.

        Args:
            text: Query text.

        Returns:
            Sparse vector as ``{dimension_index: idf_weight}``.
        """
        tokens = _uni_tokenize(text)
        tf_counts = Counter(tokens)
        sparse: Dict[int, float] = {}
        for token, tf in tf_counts.items():
            tid = self._vocab.get(token)
            if tid is None:
                continue
            df = self._df.get(tid, 0)
            if df == 0:
                continue
            idf = math.log((self._doc_count - df + 0.5) / (df + 0.5) + 1.0)
            if idf > 0:
                sparse[tid] = idf * tf
        return sparse
