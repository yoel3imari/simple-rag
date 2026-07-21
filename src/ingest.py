import json
import shutil
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

import quiver_vector_db as quiver
from src.bm25 import UnicodeBM25

from src.config import (
    BM25_B,
    BM25_K1,
    BM25_STATE_PATH,
    CHUNK_OVERLAP,
    CHUNK_OVERLAP_PDF,
    CHUNK_SEPARATORS,
    CHUNK_SEPARATORS_PDF,
    CHUNK_SIZE,
    CHUNK_SIZE_PDF,
    EMBED_BATCH_SIZE,
    EMBEDDING_DIM,
    INGEST_STATE_PATH,
    QUIVER_STORE_PATH,
)
from src.embed import embed
from src.extract import extract_text
from src.scrape import ScrapeError, scrape_url


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


def chunk_text(
    text: str,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    separators: list[str] | None = None,
) -> list[str]:
    """Recursively split *text* using the separator hierarchy, then add overlap."""
    if not text:
        return []

    if separators is None:
        separators = list(CHUNK_SEPARATORS)
    base = _recursive_split(text, separators, size)
    merged = _merge_small(base, separators[0], size)

    if overlap <= 0 or len(merged) <= 1:
        return merged

    result = [merged[0]]
    for i in range(1, len(merged)):
        prev_tail = merged[i - 1][-overlap:] if len(merged[i - 1]) > overlap else merged[i - 1]
        result.append(prev_tail + merged[i])
    return result


def load_ingest_state() -> dict:
    """Load ingest state from disk, or return a fresh empty state."""
    if INGEST_STATE_PATH.exists():
        return json.loads(INGEST_STATE_PATH.read_text(encoding="utf-8"))
    return {"next_id": 0, "sources": {}}


def save_ingest_state(state: dict) -> None:
    """Persist ingest state to disk."""
    INGEST_STATE_PATH.write_text(json.dumps(state), encoding="utf-8")


def _init_store(recreate: bool) -> tuple[quiver.Collection, UnicodeBM25]:
    """Initialise Quiver store and UnicodeBM25, wiping if *recreate* is set."""
    QUIVER_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if recreate:
        shutil.rmtree(QUIVER_STORE_PATH, ignore_errors=True)
        BM25_STATE_PATH.unlink(missing_ok=True)
        INGEST_STATE_PATH.unlink(missing_ok=True)

    client = quiver.Client(path=str(QUIVER_STORE_PATH))

    if recreate or not QUIVER_STORE_PATH.exists():
        col = client.create_collection("docs", dimensions=EMBEDDING_DIM, metric="cosine")
    else:
        col = client.get_collection("docs")

    bm25 = UnicodeBM25(k1=BM25_K1, b=BM25_B)
    if BM25_STATE_PATH.exists():
        bm25 = UnicodeBM25.load(str(BM25_STATE_PATH))
    return col, bm25


def _get_chunk_params(suffix: str) -> tuple[int, int, list[str]]:
    """Return (size, overlap, separators) based on file extension."""
    if suffix == ".pdf":
        return CHUNK_SIZE_PDF, CHUNK_OVERLAP_PDF, list(CHUNK_SEPARATORS_PDF)
    return CHUNK_SIZE, CHUNK_OVERLAP, list(CHUNK_SEPARATORS)


def _handle_existing_document(state: dict, name: str, col: quiver.Collection, bm25: UnicodeBM25, gid: int) -> None:
    """Remove existing chunks for a document being force re-ingested."""
    if name in state["sources"]:
        start, end = state["sources"][name]
        for i in range(start, end):
            col.delete(i)
            bm25.remove_document(i)


def _embed_chunks(chunks: list[str]) -> np.ndarray:
    """Embed chunks in batches and return normalised vectors.

    Raises:
        RuntimeError: If any embedding batch fails.
    """
    all_vectors: list[np.ndarray] = []
    for b in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[b : b + EMBED_BATCH_SIZE]
        try:
            vecs = embed(batch)
        except Exception as e:
            raise RuntimeError(f"Embedding failed for batch {b // EMBED_BATCH_SIZE}: {e}") from e
        all_vectors.append(vecs)
    vectors = np.concatenate(all_vectors, axis=0)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def _upsert_chunks(
    chunks: list[str],
    vectors: np.ndarray,
    col: quiver.Collection,
    bm25: UnicodeBM25,
    gid: int,
    name: str,
    lang: str,
    *,
    extra_payload: dict | None = None,
) -> None:
    """Upsert chunks with hybrid (dense + sparse) vectors into the collection.

    Raises:
        RuntimeError: If any upsert operation fails.
    """
    for i, chunk in enumerate(chunks):
        vid = gid + i
        try:
            sparse = bm25.index_document(vid, chunk)
            payload = {
                "text": chunk,
                "source": name,
                "lang": lang,
                "chunk_index": vid,
            }
            if extra_payload:
                payload.update(extra_payload)
            col.upsert_hybrid(
                id=vid,
                vector=vectors[i].tolist(),
                sparse_vector=sparse or None,
                payload=payload,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to upsert chunk {vid} for '{name}': {e}") from e


def _ingest_web(
    url: str,
    col: quiver.Collection,
    bm25: UnicodeBM25,
    state: dict,
    force: bool = False,
) -> int:
    """Ingest a single URL — scrape, chunk, embed, and index."""
    name = url

    if name in state["sources"] and not force:
        print(f"[ingest] '{name}' already ingested, skipping (use --force to re-ingest)")
        return 0

    try:
        markdown = scrape_url(url)
    except ScrapeError as e:
        print(f"[ingest] Skipping {url}: {e.message}")
        return 0

    text = markdown
    lang = "en"

    gid = state["next_id"]
    _handle_existing_document(state, name, col, bm25, gid)

    size, overlap, seps = _get_chunk_params(".txt")
    chunks = chunk_text(text, size=size, overlap=overlap, separators=seps)
    print(f"[ingest] {name}: {len(chunks)} chunks (size={size}, overlap={overlap})")

    if not chunks:
        return 0

    extra_payload = {"source_url": url, "source_type": "web"}

    try:
        vectors = _embed_chunks(chunks)
        _upsert_chunks(chunks, vectors, col, bm25, gid, name, lang, extra_payload=extra_payload)
    except RuntimeError as e:
        print(f"[ingest] {e}")
        return 0

    state["next_id"] = gid + len(chunks)
    state["sources"][name] = [gid, gid + len(chunks)]
    save_ingest_state(state)
    bm25.save(str(BM25_STATE_PATH))

    print(f"[ingest] Indexed {len(chunks)} vectors from {name}")
    return len(chunks)


def _ingest_file(
    path: str,
    col: quiver.Collection,
    bm25: UnicodeBM25,
    state: dict,
    force: bool = False,
) -> int:
    """Ingest a single local file — extract, chunk, embed, and index."""
    p = Path(path)
    if not p.exists():
        print(f"[ingest] File not found: {path}")
        return 0

    try:
        text, lang = extract_text(str(p))
    except ValueError as e:
        print(f"[ingest] Skipping {p.name}: {e}")
        return 0

    if not text.strip():
        print(f"[ingest] No text extracted from '{p.name}', skipping")
        return 0

    name = p.name

    if name in state["sources"] and not force:
        print(f"[ingest] '{name}' already ingested, skipping (use --force to re-ingest)")
        return 0

    size, overlap, seps = _get_chunk_params(p.suffix.lower())
    gid = state["next_id"]
    _handle_existing_document(state, name, col, bm25, gid)

    chunks = chunk_text(text, size=size, overlap=overlap, separators=seps)
    print(f"[ingest] {name}: {len(chunks)} chunks (size={size}, overlap={overlap})")

    if not chunks:
        return 0

    try:
        vectors = _embed_chunks(chunks)
        _upsert_chunks(chunks, vectors, col, bm25, gid, name, lang)
    except RuntimeError as e:
        print(f"[ingest] {e}")
        return 0

    state["next_id"] = gid + len(chunks)
    state["sources"][name] = [gid, gid + len(chunks)]
    save_ingest_state(state)
    bm25.save(str(BM25_STATE_PATH))

    print(f"[ingest] Indexed {len(chunks)} vectors from {name}")
    return len(chunks)


def ingest_document(
    path: str,
    col: quiver.Collection,
    bm25: UnicodeBM25,
    state: dict,
    force: bool = False,
) -> int:
    """Ingest a single document — dispatches to URL or file path."""
    if urlparse(path).scheme in ("http", "https"):
        return _ingest_web(path, col, bm25, state, force=force)
    return _ingest_file(path, col, bm25, state, force=force)


def ingest(sources: str | list[str], recreate: bool = False, force: bool = False) -> int:
    """Ingest one or more documents into the shared collection.

    Args:
        sources: Single file path or list of file paths.
        recreate: If True, wipe existing store and start fresh.
        force: If True, re-ingest already-ingested documents.

    Returns:
        Total number of vectors indexed.
    """
    if isinstance(sources, str):
        sources = [sources]

    col, bm25 = _init_store(recreate)
    state = {"next_id": 0, "sources": {}} if recreate else load_ingest_state()

    total = 0
    for src in sources:
        total += ingest_document(src, col, bm25, state, force=force)

    print(f"[ingest] Total: {total} vectors indexed across {len(sources)} source(s)")
    return total
