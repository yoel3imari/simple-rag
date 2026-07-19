# simple-rag

Step-by-step RAG learning project. Start minimal, scale progressively.

## Stack

- **Embeddings**: `nomic-embed-text` via Ollama (768d, no PyTorch)
- **Vector search**: Quiver `FlatIndex` (cosine, AVX2-accelerated)
- **Keyword search**: Quiver BM25 (hybrid fusion via weighted score)
- **Re-ranker**: Cross-encoder `ms-marco-MiniLM-L-6-v2` via ONNX Runtime
- **Generation**: `gemma3` via Ollama
- **Chunking**: Recursive splitter with overlap (512 chars, 64 overlap)

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) running locally

## Setup

```bash
# Install Python deps
uv sync

# Pull models
ollama pull nomic-embed-text
ollama pull gemma3
```

## Usage

```bash
# Ingest a document
python -m src.main ingest data/sample.txt

# Re-ingest from scratch (wipes existing index)
python -m src.main ingest data/sample.txt --recreate

# Ask a question
python -m src.main query "What are the failure modes of RAG?"
```

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                     CLI (main.py)                     │
│    python -m src.main ingest <file>                  │
│    python -m src.main query "<question>"              │
└────────┬─────────────────────────────┬───────────────┘
         │                             │
         ▼                             ▼
┌─────────────────┐       ┌─────────────────────────────────────┐
│    ingest.py     │       │            query.py                  │
│                  │       │                                      │
│  read file       │       │  embed question                      │
│  recursive split │       │  Quiver search_hybrid():             │
│  embed chunks    │       │  ┌─ Cosine (FlatIndex)              │
│  → Quiver col    │       │  ├─ BM25 keyword search             │
│  → BM25 state    │       │  └─ Weighted fusion → top-k         │
│                  │       │  → rerank.py (cross-encoder ONNX)    │
│                  │       │  → build prompt                     │
│                  │       │  → Ollama generate                   │
│                  │       │  → return answer                     │
└────────┬─────────┘       └────────┬────────────────────────────┘
         │                          │
         ▼                          ▼
┌──────────────────────────────────────────┐
│              embed.py                     │
│  POST /api/embed → np.ndarray            │
└──────────────────────────────────────────┘
         │                          │
         ▼                          ▼
┌──────────────────────────────────────────┐
│              store/                       │
│  quiver/     │  bm25.json                │
│  (collection)│  (BM25 state)             │
└──────────────────────────────────────────┘
```

## Project structure

```
simple-rag/
├── src/
│   ├── __init__.py
│   ├── config.py      # All tunable parameters
│   ├── embed.py       # Ollama embedding client
│   ├── ingest.py      # Document ingestion pipeline
│   ├── query.py       # Retrieval + re-ranking + generation
│   ├── rerank.py      # Cross-encoder re-ranker (ONNX)
│   └── main.py        # CLI dispatcher
├── data/
│   └── sample.txt     # Test document
├── store/             # Quiver collection + BM25 state (gitignored)
├── pyproject.toml
└── README.md
```

## Roadmap

| # | What | Why |
|---|---|---|
| v1 | ✅ Fixed chunks, FAISS, basic prompt | Baseline |
| v2 | ✅ Recursive/semantic chunking | Fix broken context windows |
| v3 | ✅ Hybrid search (FAISS + BM25) | Catch exact-match misses |
| v4 | ✅ Re-ranking with cross-encoder | Sharper top-k ordering |
| v5 | Evaluation (hit rate, faithfulness) | Stop guessing if changes help |
| v6 | Query rewriting / HyDE | Better retrieval for ambiguous queries |
| v7 | Multi-document / PDF / web sources | Real-world ingestion |
