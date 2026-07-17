# simple-rag

Step-by-step RAG learning project. Start minimal, scale progressively.

## Stack

- **Embeddings**: `nomic-embed-text` via Ollama (768d, no PyTorch)
- **Vector store**: FAISS `IndexFlatIP` (cosine similarity, exact search, CPU)
- **Generation**: `gemma3` via Ollama
- **Chunking**: Fixed-size (512 chars, 64 overlap) — deliberately naive

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
┌─────────────────┐       ┌─────────────────────────┐
│    ingest.py     │       │        query.py          │
│                  │       │                          │
│  read file       │       │  embed question          │
│  chunk text      │       │  → FAISS search (top-k)  │
│  embed chunks    │       │  → build prompt          │
│  → FAISS index   │       │  → Ollama generate       │
│  → metadata.pkl  │       │  → return answer         │
└────────┬─────────┘       └────────┬─────────────────┘
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
│  index.faiss  │  metadata.pkl            │
│  (FAISS IVF)  │  (chunk text + source)   │
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
│   ├── query.py       # Retrieval + generation
│   └── main.py        # CLI dispatcher
├── data/
│   └── sample.txt     # Test document
├── store/             # FAISS index + metadata (gitignored)
├── pyproject.toml
└── README.md
```

## Roadmap

| # | What | Why |
|---|---|---|
| v1 | ✅ Current: fixed chunks, FAISS, basic prompt | Baseline |
| v2 | Recursive/semantic chunking | Fix broken context windows |
| v3 | Hybrid search (FAISS + BM25) | Catch exact-match misses |
| v4 | Re-ranking with cross-encoder | Sharper top-k ordering |
| v5 | Evaluation (hit rate, faithfulness) | Stop guessing if changes help |
| v6 | Query rewriting / HyDE | Better retrieval for ambiguous queries |
| v7 | Multi-document / PDF / web sources | Real-world ingestion |
