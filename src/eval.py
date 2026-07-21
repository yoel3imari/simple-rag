import json
import unicodedata
from pathlib import Path

import numpy as np

import quiver_vector_db as quiver
from src.bm25 import UnicodeBM25
from src.prompts import build_generation_prompt

from src.config import (
    BM25_B,
    BM25_K1,
    BM25_STATE_PATH,
    EVAL_QUERIES_PATH,
    OLLAMA_GEN_MODEL,
    QUIVER_STORE_PATH,
    RERANKER_MODEL,
    TOP_K,
    TOP_K_RERANK,
)
from src.llm import ollama_chat
from src.query_transform import (
    should_transform,
    detect_language,
    rewrite_query,
    generate_hypothetical_doc,
    run_hyde_fusion,
    _standard_retrieve,
)
from src.constants import TransformMode
from src.config import QUERY_TRANSFORM_MODE


# ── Normalisation helper ──────────────────────────────────────────


def norm(text: str) -> str:
    """NFKD normalise, strip combining marks, lowercase, collapse whitespace."""
    t = unicodedata.normalize("NFKD", text)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.lower()
    return " ".join(t.split())


# ── Dataset ──────────────────────────────────────────────────────


def load_dataset(path: str | Path = EVAL_QUERIES_PATH) -> list[dict]:
    with open(path) as f:
        return json.load(f)


# ── Retrieval pipeline (shared with query.py) ────────────────────


def _load_store():
    client = quiver.Client(path=str(QUIVER_STORE_PATH))
    col = client.get_collection("docs")
    bm25 = UnicodeBM25.load(str(BM25_STATE_PATH))
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
    embed_text = question

    if mode in (TransformMode.REWRITE, TransformMode.HYDE, TransformMode.HYDE_FUSION) and should_transform(question):
        lang = detect_language(question)
        if mode == TransformMode.REWRITE:
            embed_text = rewrite_query(question, lang)
        elif mode == TransformMode.HYDE:
            embed_text = generate_hypothetical_doc(question, lang)

    if mode == TransformMode.HYDE_FUSION and should_transform(question):
        hits, _ = run_hyde_fusion(
            question, col, bm25,
            top_k=top_k, use_rerank=use_rerank, use_bm25=use_bm25,
        )
        return hits

    hits, _ = _standard_retrieve(
        question, embed_text, col, bm25,
        top_k=top_k, top_k_rerank=top_k_rerank,
        use_rerank=use_rerank, use_bm25=use_bm25,
    )
    return hits


# ── Metrics ──────────────────────────────────────────────────────


def _hit_at_k(retrieved_texts: list[str], expected: list[str]) -> bool:
    """Did any expected substring appear in any retrieved chunk?"""
    if not expected:
        return True  # negative query
    norm_exp = [norm(e) for e in expected]
    return any(any(ne in norm(rt) for rt in retrieved_texts) for ne in norm_exp)


def _recall_at_k(retrieved_texts: list[str], expected: list[str]) -> float:
    """Fraction of expected substrings covered by retrieved chunks."""
    if not expected:
        return 1.0
    norm_exp = [norm(e) for e in expected]
    covered = [ne for ne in norm_exp if any(ne in norm(rt) for rt in retrieved_texts)]
    return len(covered) / len(expected)


def _mrr(retrieved_texts: list[str], expected: list[str]) -> float:
    """Reciprocal rank of first hit chunk containing any expected substring."""
    if not expected:
        return 1.0
    norm_exp = [norm(e) for e in expected]
    for rank, rt in enumerate(retrieved_texts, start=1):
        if any(ne in norm(rt) for ne in norm_exp):
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
    content = ollama_chat(OLLAMA_GEN_MODEL, prompt_text, timeout=120)

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
        expected = entry.get("expected_substrings", [])

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

        retrieved_texts = [h["payload"]["text"] for h in hits]
        retrieved_ids = [h["payload"]["chunk_index"] for h in hits]

        norm_exp = [norm(e) for e in expected]
        covered = [e for e, ne in zip(expected, norm_exp) if any(ne in norm(rt) for rt in retrieved_texts)]

        row = {
            "question": question,
            "category": entry["category"],
            "expected": expected,
            "retrieved": retrieved_ids,
            "hit": _hit_at_k(retrieved_texts, expected),
            "recall": _recall_at_k(retrieved_texts, expected),
            "mrr": _mrr(retrieved_texts, expected),
            "covered": covered,
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
    modes = [TransformMode.NONE, TransformMode.REWRITE, TransformMode.HYDE, TransformMode.HYDE_FUSION]
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
    prompt = build_generation_prompt(question, context)
    return ollama_chat(OLLAMA_GEN_MODEL, prompt, timeout=120)


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
        covered_str = f" covered={r['covered']}" if r["covered"] else ""
        print(
            f"  [{status}] {r['category']:<10} expected={r['expected']}  got={r['retrieved']}"
            f"{covered_str}  recall={r['recall']:.2f}  MRR={r['mrr']:.2f}"
        )
        print(f"       {r['question'][:70]}")
