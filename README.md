# simple-rag

Step-by-step RAG learning project. Start minimal, scale progressively.

## Stack

- **Embeddings**: `nomic-embed-text` via Ollama (768d, no PyTorch)
- **Vector search**: Quiver `FlatIndex` (cosine, AVX2-accelerated)
- **Keyword search**: Quiver BM25 (hybrid fusion via weighted score)
- **Re-ranker**: Cross-encoder `ms-marco-MiniLM-L-6-v2` via ONNX Runtime
- **Evaluation**: Hit Rate, Recall, MRR + LLM-as-judge faithfulness
- **Generation**: `gemma3` via Ollama
- **Chunking**: Recursive splitter with overlap (512 chars, 64 overlap)
- **Query transformation (v6)**: Optional query rewriting / HyDE with a heuristic gate (`should_transform`) and multilingual prompts (EN/AR/FR). Modes: `none`, `rewrite`, `hyde`, `hyde_fusion`.

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

# Evaluate retrieval quality
python -m src.main eval

# Ablation: isolate each component
python -m src.main eval --no-rerank        # hybrid search only
python -m src.main eval --dense-only       # dense vector only
python -m src.main eval --no-bm25           # dense + rerank
python -m src.main eval --faithfulness      # include generation fidelity check

# Query transformation modes (v6)
python -m src.main query "What's his background?" --transform rewrite
python -m src.main query "Summarize his career" --transform hyde
python -m src.main query "Tell me about him" --transform hyde_fusion

# Compare all transform modes on the eval set
python -m src.main eval --compare-transforms
python -m src.main eval --transform hyde
```

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                     CLI (main.py)                    │
│    python -m src.main ingest <file>                  │
│    python -m src.main query "<question>"             │
└────────┬─────────────────────────────┬───────────────┘
         │                             │
         ▼                             ▼
┌──────────────────┐       ┌─────────────────────────────────────┐
│    ingest.py     │       │            query.py                 │
│                  │       │                                     │
│  read file       │       │  [gate] should_transform?           │
│  recursive split │       │   ├─ no  → embed raw query          │
│  embed chunks    │       │   └─ yes → rewrite / HyDE           │
│  → Quiver col    │       │          → embed transformed        │
│  → BM25 state    │       │  Quiver search_hybrid():            │
│                  │       │  ┌─ Cosine (FlatIndex)              │
│                  │       │  ├─ BM25 keyword search             │
│                  │       │  └─ Weighted fusion → top-k         │
│                  │       │  → rerank.py (cross-encoder ONNX,   │
│                  │       │       uses ORIGINAL question)       │
│                  │       │  → build prompt                     │
│                  │       │  → Ollama generate                  │
│                  │       │  → return answer                    │
└────────┬─────────┘       └────────┬────────────────────────────┘
         │                          │
         ▼                          ▼
┌──────────────────────────────────────────┐
│              embed.py                    │
│  POST /api/embed → np.ndarray            │
└──────────────────────────────────────────┘
         │                          │
         ▼                          ▼
┌──────────────────────────────────────────┐
│              store/                      │
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
│   ├── query_transform.py  # v6: query rewriting / HyDE, gate, multilingual prompts, RRF fusion
│   ├── rerank.py      # Cross-encoder re-ranker (ONNX)
│   ├── eval.py        # Evaluation metrics + runner
│   └── main.py        # CLI dispatcher
├── data/
│   ├── sample.txt     # Test document
│   └── eval_queries.json  # Ground-truth dataset (23 queries, with lang field)
├── store/             # Quiver collection + BM25 state (gitignored)
├── pyproject.toml
└── README.md
```

## Query Transformation (v6)

Before embedding, the query can be transformed to bridge the gap between how
users ask questions and how documents are written. A heuristic gate
(`should_transform`) skips well-formed queries (≥5 words + a named entity) to
avoid degrading factoid retrieval — transformation only fires on short, vague,
or cross-lingual queries.

Four modes (set via `--transform` or `QUERY_TRANSFORM_MODE` in `config.py`):

| Mode | What happens | Best for |
|------|--------------|----------|
| `none` | Raw query embedded directly | Well-formed factoids (default) |
| `rewrite` | LLM rewrites query into a keyword-rich search form | Short / vague / cross-lingual |
| `hyde` | LLM hallucinates a hypothetical answer passage, embed that | Abstract / conceptual |
| `hyde_fusion` | Run raw + HyDE, fuse rankings via Reciprocal Rank Fusion (RRF) | Ambiguous / multi-angle |

The cross-encoder re-ranker **always** receives the original question, never the
transformed text. Prompts are selected per language (English / Arabic / French)
based on query content.

Compare modes on the eval set:

```bash
python -m src.main eval --compare-transforms
```

## Roadmap

| # | What | Why |
|---|---|---|
| v1 | ✅ Fixed chunks, brute-force cosine, basic prompt | Baseline |
| v2 | ✅ Recursive/semantic chunking | Fix broken context windows |
| v3 | ✅ Hybrid search (cosine + BM25) | Catch exact-match misses |
| v4 | ✅ Re-ranking with cross-encoder | Sharper top-k ordering |
| v5 | ✅ Evaluation (hit rate, recall, MRR, faithfulness) | Stop guessing if changes help |
| v6 | ✅ Query rewriting / HyDE (gated, multilingual, RRF fusion) | Better retrieval for ambiguous / short / cross-lingual queries |
| v7 | Multi-document / PDF / web sources | Real-world ingestion |
