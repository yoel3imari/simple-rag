import os
from functools import cache

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from src.config import RERANKER_MODEL

_REPO_ID = "Xenova/ms-marco-MiniLM-L-6-v2"
_MODEL_FILES = ["onnx/model.onnx", "tokenizer.json", "config.json"]


def _snapshot_dir() -> str:
    """Download cross-encoder ONNX files and return the snapshot directory."""
    for filename in _MODEL_FILES:
        hf_hub_download(repo_id=_REPO_ID, filename=filename)
    # Use a root-level file to find the snapshot dir
    root_file = hf_hub_download(repo_id=_REPO_ID, filename=_MODEL_FILES[1])
    return os.path.dirname(root_file)


@cache
def _load_pipeline():
    """Lazy-load cross-encoder tokenizer + ONNX session (cached)."""
    snapshot = _snapshot_dir()

    tokenizer = Tokenizer.from_file(os.path.join(snapshot, "tokenizer.json"))
    tokenizer.enable_truncation(max_length=512)

    session = ort.InferenceSession(
        os.path.join(snapshot, "onnx/model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    output_name = session.get_outputs()[0].name

    return tokenizer, session, output_name


def rerank(query: str, texts: list[str]) -> list[float]:
    """Score each (query, text) pair with a cross-encoder (ONNX).

    Returns list of relevance scores in the same order as *texts*.
    Higher score = more relevant.
    """
    tokenizer, session, output_name = _load_pipeline()

    if not texts:
        return []

    # Batch-encode all (query, text) pairs with padding to longest
    pairs = [(query, t) for t in texts]
    re_encode = tokenizer.encode

    # First pass to find max length
    max_len = max(len(re_encode(q, t).ids) for q, t in pairs)

    tokenizer.enable_padding(pad_id=0, pad_type_id=0, length=max_len)
    batch_ids = np.array([re_encode(q, t).ids for q, t in pairs], dtype=np.int64)
    batch_mask = np.array(
        [re_encode(q, t).attention_mask for q, t in pairs], dtype=np.int64
    )
    batch_type = np.array(
        [re_encode(q, t).type_ids for q, t in pairs], dtype=np.int64
    )

    scores = session.run(
        [output_name],
        {
            "input_ids": batch_ids,
            "attention_mask": batch_mask,
            "token_type_ids": batch_type,
        },
    )[0]

    return scores.flatten().tolist()
