from pathlib import Path

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
OLLAMA_BASE_URL = "http://localhost:11434"
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
TOP_K_RERANK = 10      # candidate pool size before re-ranking
