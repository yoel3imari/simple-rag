# simple-rag

Step-by-step RAG learning project. Start minimal, scale progressively.

## Stack

- **Embeddings**: `nomic-embed-text` via Ollama (768d, no PyTorch)
- **Vector search**: Quiver `FlatIndex` (cosine, AVX2-accelerated)
- **Keyword search**: Quiver BM25 (hybrid fusion via weighted score)
- **Re-ranker**: Cross-encoder `ms-marco-MiniLM-L-6-v2` via ONNX Runtime
- **Evaluation**: Hit Rate, Recall, MRR + LLM-as-judge faithfulness
- **Generation**: `gemma3` via Ollama
- **Chunking**: Recursive splitter with overlap (512 chars / 64 overlap for text; 1024 chars / 128 overlap for PDF)
- **Query transformation (v6)**: Optional query rewriting / HyDE with a heuristic gate (`should_transform`) and multilingual prompts (EN/AR/FR). Modes: `none`, `rewrite`, `hyde`, `hyde_fusion`.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) running locally
- [Tesseract](https://github.com/tesseract-ocr/tesseract) (for PDF OCR fallback, optional)

```bash
# macOS
brew install tesseract tesseract-lang

# Download Arabic tessdata (required for ara+eng OCR)
mkdir -p /tmp/tessdata
curl -L -o /tmp/tessdata/ara.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/ara.traineddata
export TESSDATA_PREFIX=/tmp/tessdata
```

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

# Multi-document ingest (v7)
python -m src.main ingest data/sample.txt data/ma064fr.pdf data/ma064ar.pdf

# Ingest an entire directory (recursively, .txt + .pdf)
python -m src.main ingest data/ --force

# Check what's been ingested
python -m src.main ingest --status
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
│  extract.py      │       │  [gate] should_transform?           │
│   ├─ .txt read   │       │   ├─ no  → embed raw query          │
│   ├─ .pdf (PyMuPDF)      │   └─ yes → rewrite / HyDE           │
│   └─ OCR fallback│       │          → embed transformed        │
│  recursive split │       │  Quiver search_hybrid():            │
│  embed chunks    │       │  ┌─ Cosine (FlatIndex)              │
│  → Quiver col    │       │  ├─ BM25 keyword search             │
│  → BM25 state    │       │  └─ Weighted fusion → top-k         │
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
│              │  ingest_state.json         │
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
│   ├── extract.py     # PDF/TXT extraction + language detection + OCR fallback
│   ├── bm25.py        # Unicode-aware BM25 subclass (Arabic + French support)
│   ├── query_transform.py  # v6: query rewriting / HyDE, gate, multilingual prompts, RRF fusion
│   ├── rerank.py      # Cross-encoder re-ranker (ONNX)
│   ├── eval.py        # Evaluation metrics + runner
│   └── main.py        # CLI dispatcher
├── data/
│   ├── sample.txt     # Test document
│   └── eval_queries.json  # Ground-truth dataset (29 queries, with lang field)
├── store/             # Quiver collection + BM25 state + ingest state (gitignored)
│   ├── quiver/        # Vector collection
│   ├── bm25.json      # BM25 state
│   └── ingest_state.json  # Per-file ingest tracking
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

## Multi-Document & PDF Support (v7)

v7 adds first-class support for multi-file ingestion, PDF text extraction with
OCR fallback, and per-file state tracking.

**Multi-file and directory ingest.** Pass multiple files on one command, or
point at a directory. The CLI recursively discovers `.txt` and `.pdf` files
under the given path:

```bash
# Multiple files in one go
python -m src.main ingest data/sample.txt data/ma064fr.pdf data/ma064ar.pdf

# Entire directory
python -m src.main ingest data/
```

**PDF extraction.** Text is extracted via PyMuPDF (`fitz`). When a PDF has
garbled text (broken ToUnicode CMap, common in Arabic legal PDFs), the system
detects the newline-per-glyph pattern and falls back to OCR.

**OCR fallback.** Each PDF page is rendered to PNG at 200 DPI, then passed
to Tesseract with `ara+eng` language models. Requires `TESSDATA_PREFIX` set
to a directory containing `ara.traineddata`.

**PDF chunking.** PDFs use larger chunks (1024 chars / 128 overlap) to preserve
sentence-level context in dense legal text. Text files keep the original 512/64
settings. Separator lists also differ: PDF drops `. ` (unreliable for Arabic
and French sentence boundaries).

**Ingest state tracking.** A JSON file (`store/ingest_state.json`) records
which files have been ingested, their SHA-256 hashes, chunk counts, and
languages. This avoids re-embedding unchanged documents:

```bash
# See what's been ingested
python -m src.main ingest --status
```

**Force re-ingest.** The `--force` flag wipes the existing index and ingests
from scratch, useful after chunking or extraction changes:

```bash
python -m src.main ingest data/ --force
```

**Unicode BM25.** `UnicodeBM25` subclasses Quiver's BM25 with a tokenizer
that handles Arabic Unicode ranges and accented Latin characters. Stock BM25
uses an ASCII-only regex (`[a-z0-9]+`) that produces zero tokens for Arabic
and mangles French text.

**Language-aware re-ranker guard.** The cross-encoder re-ranker
(`ms-marco-MiniLM-L-6-v2`) is trained on English. For Arabic and French
queries, the re-ranker is skipped and the hybrid fusion ranking is returned
directly, avoiding noisy reordering.

## Roadmap

| # | What | Why |
|---|---|---|
| v1 | ✅ Fixed chunks, brute-force cosine, basic prompt | Baseline |
| v2 | ✅ Recursive/semantic chunking | Fix broken context windows |
| v3 | ✅ Hybrid search (cosine + BM25) | Catch exact-match misses |
| v4 | ✅ Re-ranking with cross-encoder | Sharper top-k ordering |
| v5 | ✅ Evaluation (hit rate, recall, MRR, faithfulness) | Stop guessing if changes help |
| v6 | ✅ Query rewriting / HyDE (gated, multilingual, RRF fusion) | Better retrieval for ambiguous / short / cross-lingual queries |
| v7 | ✅ Multi-document / PDF / ingest state / force-reingest | Real-world ingestion |
