import shutil
from pathlib import Path

import numpy as np

import quiver_vector_db as quiver
from quiver_vector_db import BM25

from src.config import (
    BM25_B,
    BM25_K1,
    BM25_STATE_PATH,
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    CHUNK_SIZE,
    EMBEDDING_DIM,
    QUIVER_STORE_PATH,
)
from src.embed import embed


def _merge_small(pieces: list[str], separator: str, size: int) -> list[str]:
    """Merge adjacent small pieces up to *size* using *separator* as joiner."""
    merged: list[str] = []
    current = ""
    for piece in pieces:
        if not piece:
            continue
        if not current:
            current = piece
        elif len(current) + len(separator) + len(piece) <= size:
            current += separator + piece
        else:
            merged.append(current)
            current = piece
    if current:
        merged.append(current)
    return merged


def _recursive_split(text: str, separators: list[str], size: int) -> list[str]:
    """Split *text* by the first separator; recurse finer for oversize pieces."""
    if len(text) <= size or not separators:
        return [text] if text else []

    sep = separators[0]

    # Character-level fallback — hard-cut every *size* chars
    if sep == "":
        return [text[i : i + size] for i in range(0, len(text), size)]

    parts: list[str] = text.split(sep)
    pieces: list[str] = []
    for part in parts:
        if not part:
            continue
        if len(part) <= size:
            pieces.append(part)
        else:
            pieces.extend(_recursive_split(part, separators[1:], size))
    return _merge_small(pieces, sep, size)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Recursively split *text* using the separator hierarchy, then add overlap."""
    if not text:
        return []

    base = _recursive_split(text, list(CHUNK_SEPARATORS), size)
    merged = _merge_small(base, CHUNK_SEPARATORS[0], size)

    if overlap <= 0 or len(merged) <= 1:
        return merged

    result = [merged[0]]
    for i in range(1, len(merged)):
        prev_tail = merged[i - 1][-overlap:] if len(merged[i - 1]) > overlap else merged[i - 1]
        result.append(prev_tail + merged[i])
    return result


def _init_store(recreate: bool) -> tuple[quiver.Collection, BM25]:
    """Initialise Quiver store and BM25, wiping if *recreate* is set."""
    QUIVER_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if recreate:
        shutil.rmtree(QUIVER_STORE_PATH, ignore_errors=True)
        BM25_STATE_PATH.unlink(missing_ok=True)

    client = quiver.Client(path=str(QUIVER_STORE_PATH))

    if recreate or not QUIVER_STORE_PATH.exists():
        col = client.create_collection("docs", dimensions=EMBEDDING_DIM, metric="cosine")
    else:
        col = client.get_collection("docs")

    bm25 = BM25(k1=BM25_K1, b=BM25_B)
    return col, bm25


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

    # 5. Initialise Quiver store + BM25
    col, bm25 = _init_store(recreate)

    # 6. Upsert vectors with payloads
    batch = [
        (i, vectors[i].tolist(), {"text": chunks[i], "source": source, "chunk_index": i})
        for i in range(len(chunks))
    ]
    col.upsert_batch(batch)
    print(f"[ingest] Indexed {len(chunks)} vectors in Quiver (collection='docs')")

    # 7. Build and persist BM25 index
    for i, chunk in enumerate(chunks):
        bm25.index_document(i, chunk)
    bm25.save(str(BM25_STATE_PATH))
    print(f"[ingest] Saved BM25 state ({BM25_STATE_PATH})")
