import pickle
from pathlib import Path

import numpy as np

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_DIM, METADATA_PATH, VECTORS_PATH
from src.embed import embed


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def ingest(file_path: str, recreate: bool = False) -> None:
    path = Path(file_path)
    if not path.exists():
        print(f"File not found: {file_path}")
        return

    # 1. Read
    text = path.read_text(encoding="utf-8")
    source = path.name
    print(f"[ingest] Loaded {len(text)} chars from {source}")

    # 2. Chunk
    chunks = chunk_text(text)
    print(f"[ingest] Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # 3. Embed
    print(f"[ingest] Embedding {len(chunks)} chunks with {EMBEDDING_DIM}-dim model...")
    vectors = embed(chunks)
    print(f"[ingest] Embedding done, shape={vectors.shape}")

    # 4. Normalize vectors for cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    # 5. Build metadata list (position in array -> chunk info)
    metadata = [
        {"text": chunk, "source": source, "chunk_index": i}
        for i, chunk in enumerate(chunks)
    ]
    print(f"[ingest] Index has {len(vectors)} vectors")

    # 6. Persist
    VECTORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if recreate:
        for p in [VECTORS_PATH, METADATA_PATH]:
            p.unlink(missing_ok=True)

    np.save(VECTORS_PATH, vectors)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    print(f"[ingest] Saved vectors ({VECTORS_PATH}) and metadata ({METADATA_PATH})")
