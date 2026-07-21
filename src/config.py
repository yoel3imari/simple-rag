import os
from pathlib import Path

from src.constants import TransformMode

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STORE_DIR = PROJECT_ROOT / "store"

# Quiver vector store
QUIVER_STORE_PATH = STORE_DIR / "quiver"
BM25_STATE_PATH = STORE_DIR / "bm25.json"

# Embedding model (lightweight, no PyTorch)
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768

# Generation model
OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_GEN_MODEL = "gemma3:1b"

# Chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Retrieval
TOP_K = 3

# Hybrid search weights (dense = vector cosine, sparse = BM25)
HYBRID_DENSE_WEIGHT = 0.5
HYBRID_SPARSE_WEIGHT = 0.5

# BM25 tuning
BM25_K1 = 1.5
BM25_B = 0.75

# Cross-encoder re-ranker
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_RERANK = 30      # candidate pool size before re-ranking (increased from 10 for better coverage with more documents)

# Evaluation
EVAL_QUERIES_PATH = DATA_DIR / "eval_queries.json"

# Transform mode: "none" | "rewrite" | "hyde" | "hyde_fusion"
QUERY_TRANSFORM_MODE: TransformMode = TransformMode.NONE

# Heuristic gating — skip transform for well-formed queries
TRANSFORM_MIN_LENGTH: int = 5             # min words to consider "well-formed"
TRANSFORM_MIN_ENTITIES: int = 1           # min named entities to consider "well-formed"

# HyDE prompts (one per language, selected by query language)
HYDE_PROMPT_EN: str = (
    "Write a 4-sentence excerpt from a document that would directly answer the "
    "question below. Use a confident, declarative tone — as if you are quoting "
    "the source. Do not hedge or express uncertainty.\n\n"
    "Question: {question}\n\nExcerpt:"
)
HYDE_PROMPT_AR: str = (
    "اكتب مقتطفًا من 4 جمل من وثيقة تجيب مباشرة على السؤال أدناه. "
    "استخدم نغمة واثقة وتقريرية — كما لو كنت تقتبس من المصدر. "
    "لا تتردد أو تعبر عن عدم اليقين.\n\n"
    "السؤال: {question}\n\nالمقتطف:"
)
HYDE_PROMPT_FR: str = (
    "Rédigez un extrait de 4 phrases d'un document qui répondrait directement "
    "à la question ci-dessous. Utilisez un ton confiant et déclaratif — comme "
    "si vous citiez la source. N'hésitez pas et n'exprimez pas d'incertitude.\n\n"
    "Question : {question}\n\nExtrait :"
)

# Rewrite prompts (one per language)
REWRITE_PROMPT_EN: str = (
    "Rewrite this question as a concise, keyword-rich search query. "
    "Include named entities and key terms. Output only the query, no explanation.\n\n"
    "Question: {question}\n\nRewritten query:"
)
REWRITE_PROMPT_AR: str = (
    "أعد كتابة هذا السؤال كاستعلام بحث موجز وغني بالكلمات المفتاحية. "
    "أدرج الكيانات والأسماء الأساسية. أخرج الاستعلام فقط دون شرح.\n\n"
    "السؤال: {question}\n\nالاستعلام:"
)
REWRITE_PROMPT_FR: str = (
    "Réécrivez cette question en une requête de recherche concise et riche en "
    "mots-clés. Incluez les entités nommées et les termes clés. Ne renvoyez que "
    "la requête, sans explication.\n\n"
    "Question : {question}\n\nRequête :"
)

# Fusion
HYDE_FUSION_TOP_K: int = 10              # candidates from each branch before fusion
HYDE_FUSION_RRF_K: int = 60              # RRF constant (standard default)

CHUNK_SIZE_PDF = 1024          # larger window for dense legal text
CHUNK_OVERLAP_PDF = 128
CHUNK_SEPARATORS_PDF = ["\n\n", "\n", " ", ""]   # drop ". " (unreliable for AR/FR)
EMBED_BATCH_SIZE = 32          # batch embed() calls (256 chunks hangs Ollama ~230s; 32 ≈30s safe)
INGEST_STATE_PATH = STORE_DIR / "ingest_state.json"
SUPPORTED_EXTENSIONS = (".txt", ".pdf")

# ── v8: Web ingestion ────────────────────────────────────────────
WEB_SCRAPER_BACKEND: str = "crawl4ai"       # "crawl4ai" | "jina"
WEB_CRAWL_MAX_PAGES: int = 50
WEB_CRAWL_MAX_DEPTH: int = 2
WEB_CRAWL_DELAY: float = 1.0                # seconds between requests
WEB_SCRAPE_TIMEOUT: int = 60                # seconds per page

# Jina Reader (optional — used when WEB_SCRAPER_BACKEND="jina")
JINA_API_KEY: str | None = None             # set via env or config; None = free tier

# Web chunking uses same constants as TXT (CHUNK_SIZE=512, CHUNK_OVERLAP=64)
