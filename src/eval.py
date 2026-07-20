import json
from pathlib import Path

import numpy as np
import requests

import quiver_vector_db as quiver
from quiver_vector_db import BM25

from src.config import (
    BM25_B,
    BM25_K1,
    BM25_STATE_PATH,
    EVAL_QUERIES_PATH,
    HYBRID_DENSE_WEIGHT,
    HYBRID_SPARSE_WEIGHT,
    OLLAMA_BASE_URL,
    OLLAMA_GEN_MODEL,
    QUIVER_STORE_PATH,
    RERANKER_MODEL,
    TOP_K,
    TOP_K_RERANK,
)
from src.embed import embed
from src.rerank import rerank
from src.query_transform import (
    should_transform,
    detect_language,
    rewrite_query,
    generate_hypothetical_doc,
    reciprocal_rank_fusion,
    RAW,
    REWRITE,
    HYDE,
    HYDE_FUSION,
)
from src.config import QUERY_TRANSFORM_MODE, HYDE_FUSION_TOP_K, HYDE_FUSION_RRF_K


# ── Dataset ──────────────────────────────────────────────────────


def load_dataset(path: str | Path = EVAL_QUERIES_PATH) -> list[dict]:
    with open(path) as f:
        return json.load(f)


# ── Retrieval pipeline (shared with query.py) ────────────────────


def _load_store():
    client = quiver.Client(path=str(QUIVER_STORE_PATH))
    col = client.get_collection("docs")
    bm25 = BM25.load(str(BM25_STATE_PATH))
    return col, bm25


def _retrieve(
    question: str,
    col,
    bm25,
    *,
    top_k: int = TOP_K,
    top_k_rerank: int = TOP_K_RERANK,
    use_rerank: bool = True,
    use_bm25: bool = True,
    transform_mode: str | None = None,
) -> list[dict]:
    """Run retrieval pipeline, return ranked list of hit dicts (payload + score).

    When *transform_mode* is provided (or set via config), applies query
    transformation (rewrite / HyDE / HyDE fusion) before embedding while
    the cross-encoder reranker always uses the **original** question.
    """
    mode = transform_mode or QUERY_TRANSFORM_MODE
    embed_text = question  # default: embed raw question

    if mode in (REWRITE, HYDE, HYDE_FUSION) and should_transform(question):
        lang = detect_language(question)
        if mode == REWRITE:
            embed_text = rewrite_query(question, lang)
        elif mode == HYDE:
            embed_text = generate_hypothetical_doc(question, lang)
        # HYDE_FUSION handled in its own block below

    # ── HYDE_FUSION: dual search + RRF fusion ──────────────────
    if mode == HYDE_FUSION and should_transform(question):
        lang = detect_language(question)
        hyde_text = generate_hypothetical_doc(question, lang)

        dense_w = HYBRID_DENSE_WEIGHT
        sparse_w = HYBRID_SPARSE_WEIGHT
        if not use_bm25:
            sparse_w = 0.0

        # Raw branch
        raw_vec = embed(question)[0]
        raw_sparse = bm25.encode_query(question)
        raw_hits = col.search_hybrid(
            dense_query=raw_vec.tolist(),
            sparse_query=raw_sparse,
            k=HYDE_FUSION_TOP_K,
            dense_weight=dense_w,
            sparse_weight=sparse_w,
        )
        # HyDE branch
        hyde_vec = embed(hyde_text)[0]
        hyde_sparse = bm25.encode_query(hyde_text)
        hyde_hits = col.search_hybrid(
            dense_query=hyde_vec.tolist(),
            sparse_query=hyde_sparse,
            k=HYDE_FUSION_TOP_K,
            dense_weight=dense_w,
            sparse_weight=sparse_w,
        )

        # Fuse by chunk_index via RRF
        raw_rank = [h["payload"]["chunk_index"] for h in raw_hits]
        hyde_rank = [h["payload"]["chunk_index"] for h in hyde_hits]
        fused = reciprocal_rank_fusion([raw_rank, hyde_rank], k=HYDE_FUSION_RRF_K)

        # Rebuild hits list from fused chunk_index order
        by_chunk = {h["payload"]["chunk_index"]: h for h in raw_hits + hyde_hits}
        hits = [by_chunk[ci] for ci, _ in fused if ci in by_chunk]

        if use_rerank and hits:
            texts = [h["payload"]["text"] for h in hits]
            ce_scores = rerank(question, texts)
            ranked = sorted(zip(hits, ce_scores), key=lambda x: x[1], reverse=True)
            hits = [h for h, _ in ranked[:top_k]]
        else:
            hits = hits[:top_k]

        return hits

    # ── Standard path (RAW, REWRITE, HYDE, and HYDE_FUSION with no transform) ──
    q_vec = embed(embed_text)[0]

    dense_w = HYBRID_DENSE_WEIGHT
    sparse_w = HYBRID_SPARSE_WEIGHT
    if not use_bm25:
        sparse_w = 0.0

    sparse_query = bm25.encode_query(question)
    k = top_k_rerank if use_rerank else top_k
    hits = col.search_hybrid(
        dense_query=q_vec.tolist(),
        sparse_query=sparse_query,
        k=k,
        dense_weight=dense_w,
        sparse_weight=sparse_w,
    )

    if use_rerank and hits:
        texts = [h["payload"]["text"] for h in hits]
        ce_scores = rerank(question, texts)
        ranked = sorted(zip(hits, ce_scores), key=lambda x: x[1], reverse=True)
        hits = [h for h, _ in ranked[:top_k]]

    return hits


# ── Metrics ──────────────────────────────────────────────────────


def _hit_at_k(retrieved: list[int], expected: list[int]) -> bool:
    """Did any expected chunk appear in retrieved list?"""
    if not expected:
        return True  # negative query: retrieving nothing is a "hit"
    return any(c in retrieved for c in expected)


def _recall_at_k(retrieved: list[int], expected: list[int]) -> float:
    """Fraction of expected chunks found in retrieved list."""
    if not expected:
        return 1.0
    return sum(1 for c in expected if c in retrieved) / len(expected)


def _mrr(retrieved: list[int], expected: list[int]) -> float:
    """Reciprocal rank of first relevant hit."""
    if not expected:
        return 1.0
    for rank, c in enumerate(retrieved, start=1):
        if c in expected:
            return 1.0 / rank
    return 0.0


# ── Faithfulness (LLM-as-judge) ──────────────────────────────────


_F_PROMPT = """You are evaluating whether a generated answer is faithful to the provided context.

Context:
{context}

Question: {question}

Generated answer:
{answer}

Rate how well the answer is supported by the context on a scale of 1 to 5:
5 = Fully supported, every claim is in the context
4 = Mostly supported, minor inference but no contradictions
3 = Partially supported, some claims not in context
2 = Poorly supported, several claims not found or contradicted
1 = Not supported, answer is unrelated or contradicts context

Return ONLY a single integer 1-5 on the first line, then a brief explanation on the second line."""


def _faithfulness_score(question: str, context: str, answer: str) -> float:
    """Ask the LLM to rate faithfulness of answer to context (1-5)."""
    if not answer or not context:
        return 0.0

    prompt_text = _F_PROMPT.format(context=context, question=question, answer=answer)
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": OLLAMA_GEN_MODEL,
            "messages": [{"role": "user", "content": prompt_text}],
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    content = resp.json()["message"]["content"].strip()

    try:
        return float(content.split("\n")[0].strip())
    except (ValueError, IndexError):
        return 0.0


# ── Full eval runner ─────────────────────────────────────────────


def run_eval(
    *,
    use_rerank: bool = True,
    use_bm25: bool = True,
    top_k: int = TOP_K,
    top_k_rerank: int = TOP_K_RERANK,
    faithfulness: bool = False,
    transform_mode: str | None = None,
) -> dict:
    """Run evaluation and return results dictionary with metrics."""
    dataset = load_dataset()
    if not dataset:
        return {"error": "Empty dataset"}

    col, bm25 = _load_store()

    rows = []
    for entry in dataset:
        question = entry["question"]
        expected = entry["answer_chunks"]

        hits = _retrieve(
            question,
            col,
            bm25,
            top_k=top_k,
            top_k_rerank=top_k_rerank,
            use_rerank=use_rerank,
            use_bm25=use_bm25,
            transform_mode=transform_mode,
        )

        retrieved = [h["payload"]["chunk_index"] for h in hits]

        row = {
            "question": question,
            "category": entry["category"],
            "expected": expected,
            "retrieved": retrieved,
            "hit": _hit_at_k(retrieved, expected),
            "recall": _recall_at_k(retrieved, expected),
            "mrr": _mrr(retrieved, expected),
        }

        if faithfulness:
            context_str = "\n\n".join(h["payload"]["text"] for h in hits)
            generated = _generate(question, context_str)
            row["generated"] = generated
            row["faithfulness"] = _faithfulness_score(question, context_str, generated)

        rows.append(row)

    return _summarise(rows)


def compare_transforms() -> dict:
    """Run eval for each transform mode and print a comparison table."""
    modes = [RAW, REWRITE, HYDE, HYDE_FUSION]
    results = {}
    for m in modes:
        print(f"\n>>> Running eval with transform_mode={m!r} ...")
        results[m] = run_eval(transform_mode=m)

    # Print comparison table
    print("\n" + "=" * 60)
    print("  TRANSFORM MODE COMPARISON")
    print("=" * 60)
    print(f"  {'Mode':<12} {'HitRate':>8} {'Recall':>8} {'MRR':>8}")
    for m in modes:
        s = results[m]
        print(f"  {m:<12} {s['hit_rate']:>8.3f} {s['recall']:>8.3f} {s['mrr']:>8.3f}")
    return results


def _generate(question: str, context: str) -> str:
    """Run the full generation pipeline (duplicated from query.py for cleanliness)."""
    if not context:
        return ""
    prompt = f"""You are a helpful assistant. Use the context below to answer the question.

Context:
{context}

Question: {question}

Answer:"""
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={"model": OLLAMA_GEN_MODEL, "messages": [{"role": "user", "content": prompt}], "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _summarise(rows: list[dict]) -> dict:
    """Aggregate per-query results into summary metrics."""
    n = len(rows)

    hits = [r["hit"] for r in rows]
    recalls = [r["recall"] for r in rows]
    mrrs = [r["mrr"] for r in rows]

    # Per-category breakdown
    categories = {}
    for r in rows:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"hits": [], "recalls": [], "mrrs": []}
        categories[cat]["hits"].append(r["hit"])
        categories[cat]["recalls"].append(r["recall"])
        categories[cat]["mrrs"].append(r["mrr"])

    for cat in categories:
        c = categories[cat]
        categories[cat] = {
            "count": len(c["hits"]),
            "hit_rate": sum(c["hits"]) / len(c["hits"]),
            "recall": sum(c["recalls"]) / len(c["recalls"]),
            "mrr": sum(c["mrrs"]) / len(c["mrrs"]),
        }

    summary = {
        "count": n,
        "hit_rate": sum(hits) / n,
        "recall": sum(recalls) / n,
        "mrr": sum(mrrs) / n,
        "categories": categories,
        "rows": rows,
    }

    if rows[0].get("faithfulness") is not None:
        faith_scores = [r["faithfulness"] for r in rows if r["faithfulness"] > 0]
        summary["faithfulness_mean"] = sum(faith_scores) / len(faith_scores) if faith_scores else 0.0
        summary["faithfulness_scores"] = faith_scores

    return summary


# ── Report ───────────────────────────────────────────────────────


def print_report(summary: dict):
    """Pretty-print evaluation results."""
    print("=" * 60)
    print(f"  EVALUATION REPORT  (n={summary['count']})")
    print("=" * 60)

    print(f"\n  Retrieval (k={TOP_K})")
    print(f"  {'─' * 40}")
    print(f"  Hit Rate@{TOP_K}    {summary['hit_rate']:.3f}")
    print(f"  Recall@{TOP_K}      {summary['recall']:.3f}")
    print(f"  MRR              {summary['mrr']:.3f}")

    if "faithfulness_mean" in summary:
        print(f"\n  Faithfulness")
        print(f"  {'─' * 40}")
        print(f"  Mean score       {summary['faithfulness_mean']:.2f} / 5.0")

    print(f"\n  By category")
    print(f"  {'─' * 40}")
    print(f"  {'Category':<15} {'Count':>6} {'HitRate':>8} {'Recall':>8} {'MRR':>8}")
    for cat, metrics in sorted(summary["categories"].items()):
        print(
            f"  {cat:<15} {metrics['count']:>6} {metrics['hit_rate']:>8.3f} {metrics['recall']:>8.3f} {metrics['mrr']:>8.3f}"
        )

    print(f"\n  Per-query details")
    print(f"  {'─' * 40}")
    for r in summary["rows"]:
        status = "✓" if r["hit"] else "✗"
        print(
            f"  [{status}] {r['category']:<10} expected={r['expected']}  got={r['retrieved']}  "
            f"recall={r['recall']:.2f}  MRR={r['mrr']:.2f}"
        )
        print(f"       {r['question'][:70]}")
