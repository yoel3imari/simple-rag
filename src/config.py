from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STORE_DIR = PROJECT_ROOT / "store"

VECTORS_PATH = STORE_DIR / "vectors.npy"
METADATA_PATH = STORE_DIR / "metadata.pkl"

# Ollama embedding model (lightweight, no PyTorch)
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768

# Ollama generation model
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_GEN_MODEL = "gemma3:1b"

# Chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# Retrieval
TOP_K = 3
